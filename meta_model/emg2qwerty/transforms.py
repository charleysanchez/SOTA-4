# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
import torch
import torchaudio


TTransformIn = TypeVar("TTransformIn")
TTransformOut = TypeVar("TTransformOut")
Transform = Callable[[TTransformIn], TTransformOut]


@dataclass
class ToTensor:
    """Extracts the specified ``fields`` from a numpy structured array
    and stacks them into a ``torch.Tensor``.

    Following TNC convention as a default, the returned tensor is of shape
    (time, field/batch, electrode_channel).

    Args:
        fields (list): List of field names to be extracted from the passed in
            structured numpy ndarray.
        stack_dim (int): The new dimension to insert while stacking
            ``fields``. (default: 1)
    """

    fields: Sequence[str] = ("emg_left", "emg_right")
    stack_dim: int = 1

    def __call__(self, data: np.ndarray) -> torch.Tensor:
        return torch.stack(
            [torch.as_tensor(data[f]) for f in self.fields], dim=self.stack_dim
        )


@dataclass
class Lambda:
    """Applies a custom lambda function as a transform.

    Args:
        lambd (lambda): Lambda to wrap within.
    """

    lambd: Transform[Any, Any]

    def __call__(self, data: Any) -> Any:
        return self.lambd(data)


@dataclass
class ForEach:
    """Applies the provided ``transform`` over each item of a batch
    independently. By default, assumes the input is of shape (T, N, ...).

    Args:
        transform (Callable): The transform to apply to each batch item of
            the input tensor.
        batch_dim (int): The bach dimension, i.e., the dim along which to
            unstack/unbind the input tensor prior to mapping over
            ``transform`` and restacking. (default: 1)
    """

    transform: Transform[torch.Tensor, torch.Tensor]
    batch_dim: int = 1

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [self.transform(t) for t in tensor.unbind(self.batch_dim)],
            dim=self.batch_dim,
        )


@dataclass
class Compose:
    """Compose a chain of transforms.

    Args:
        transforms (list): List of transforms to compose.
    """

    transforms: Sequence[Transform[Any, Any]]

    def __call__(self, data: Any) -> Any:
        for transform in self.transforms:
            data = transform(data)
        return data


@dataclass
class RandomBandRotation:
    """Applies band rotation augmentation by shifting the electrode channels
    by an offset value randomly chosen from ``offsets``. By default, assumes
    the input is of shape (..., C).

    NOTE: If the input is 3D with batch dim (TNC), then this transform
    applies band rotation for all items in the batch with the same offset.
    To apply different rotations each batch item, use the ``ForEach`` wrapper.

    Args:
        offsets (list): List of integers denoting the offsets by which the
            electrodes are allowed to be shift. A random offset from this
            list is chosen for each application of the transform.
        channel_dim (int): The electrode channel dimension. (default: -1)
    """

    offsets: Sequence[int] = (-1, 0, 1)
    channel_dim: int = -1

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        offset = np.random.choice(self.offsets) if len(self.offsets) > 0 else 0
        return tensor.roll(offset, dims=self.channel_dim)


@dataclass
class TemporalAlignmentJitter:
    """Applies a temporal jittering augmentation that randomly jitters the
    alignment of left and right EMG data by up to ``max_offset`` timesteps.
    The input must be of shape (T, ...).

    Args:
        max_offset (int): The maximum amount of alignment jittering in terms
            of number of timesteps.
        stack_dim (int): The dimension along which the left and right data
            are stacked. See ``ToTensor()``. (default: 1)
    """

    max_offset: int
    stack_dim: int = 1

    def __post_init__(self) -> None:
        assert self.max_offset >= 0

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        assert tensor.shape[self.stack_dim] == 2
        left, right = tensor.unbind(self.stack_dim)

        offset = np.random.randint(-self.max_offset, self.max_offset + 1)
        if offset > 0:
            left = left[offset:]
            right = right[:-offset]
        if offset < 0:
            left = left[:offset]
            right = right[-offset:]

        return torch.stack([left, right], dim=self.stack_dim)


@dataclass
class LogSpectrogram:
    """Creates log10-scaled spectrogram from an EMG signal. In the case of
    multi-channeled signal, the channels are treated independently.
    The input must be of shape (T, ...) and the returned spectrogram
    is of shape (T, ..., freq).

    Args:
        n_fft (int): Size of FFT, creates n_fft // 2 + 1 frequency bins.
            (default: 64)
        hop_length (int): Number of samples to stride between consecutive
            STFT windows. (default: 16)
    """

    n_fft: int = 64
    hop_length: int = 16

    def __post_init__(self) -> None:
        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            normalized=True,
            # Disable centering of FFT windows to avoid padding inconsistencies
            # between train and test (due to differing window lengths), as well
            # as to be more faithful to real-time/streaming execution.
            center=False,
        )

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        x = tensor.movedim(0, -1)  # (T, ..., C) -> (..., C, T)
        spec = self.spectrogram(x)  # (..., C, freq, T)
        logspec = torch.log10(spec + 1e-6)  # (..., C, freq, T)
        return logspec.movedim(-1, 0)  # (T, ..., C, freq)


@dataclass
class SpecAugment:
    """Applies time and frequency masking as per the paper
    "SpecAugment: A Simple Data Augmentation Method for Automatic Speech
    Recognition, Park et al" (https://arxiv.org/abs/1904.08779).

    Args:
        n_time_masks (int): Maximum number of time masks to apply,
            uniformly sampled from 0. (default: 0)
        time_mask_param (int): Maximum length of each time mask,
            uniformly sampled from 0. (default: 0)
        iid_time_masks (int): Whether to apply different time masks to
            each band/channel (default: True)
        n_freq_masks (int): Maximum number of frequency masks to apply,
            uniformly sampled from 0. (default: 0)
        freq_mask_param (int): Maximum length of each frequency mask,
            uniformly sampled from 0. (default: 0)
        iid_freq_masks (int): Whether to apply different frequency masks to
            each band/channel (default: True)
        mask_value (float): Value to assign to the masked columns (default: 0.)
    """

    n_time_masks: int = 0
    time_mask_param: int = 0
    iid_time_masks: bool = True
    n_freq_masks: int = 0
    freq_mask_param: int = 0
    iid_freq_masks: bool = True
    mask_value: float = 0.0

    def __post_init__(self) -> None:
        self.time_mask = torchaudio.transforms.TimeMasking(
            self.time_mask_param, iid_masks=self.iid_time_masks
        )
        self.freq_mask = torchaudio.transforms.FrequencyMasking(
            self.freq_mask_param, iid_masks=self.iid_freq_masks
        )

    def __call__(self, specgram: torch.Tensor) -> torch.Tensor:
        # (T, ..., C, freq) -> (..., C, freq, T)
        x = specgram.movedim(0, -1)

        # Time masks
        n_t_masks = np.random.randint(self.n_time_masks + 1)
        for _ in range(n_t_masks):
            x = self.time_mask(x, mask_value=self.mask_value)

        # Frequency masks
        n_f_masks = np.random.randint(self.n_freq_masks + 1)
        for _ in range(n_f_masks):
            x = self.freq_mask(x, mask_value=self.mask_value)

        # (..., C, freq, T) -> (T, ..., C, freq)
        return x.movedim(-1, 0)


### NEW DATA AUGMENTATION
@dataclass
class GaussianNoise:
    """Adds Gaussian noise to the EMG signal.
    
    This simulates the noise that might be present in real-world EMG recordings
    due to electronic interference, movement artifacts, etc.
    
    Args:
        mean (float): Mean of the Gaussian noise to add. (default: 0.0)
        std_range (tuple): Range of standard deviation values to sample from.
            The actual std will be randomly chosen from this range for each call.
            (default: (0.01, 0.05))
    """
    
    mean: float = 0.0
    std_range: tuple[float, float] = (0.01, 0.05)
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        std = np.random.uniform(self.std_range[0], self.std_range[1])
        noise = torch.randn_like(tensor) * std + self.mean
        return tensor + noise


@dataclass
class RandomScaling:
    """Applies random amplitude scaling to the EMG signal.
    
    This simulates variations in signal strength due to electrode placement,
    skin conductivity differences, etc.
    
    Args:
        scale_range (tuple): Range of scaling factors to sample from.
            (default: (0.8, 1.2))
    """
    
    scale_range: tuple[float, float] = (0.8, 1.2)
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        return tensor * scale


@dataclass
class ChannelDropout:
    """Randomly drops out (zeros) entire electrode channels.
    
    This simulates electrode failures or temporary connection issues.
    
    Args:
        dropout_prob (float): Probability of dropping each channel independently.
            (default: 0.1)
        channel_dim (int): The electrode channel dimension. (default: -1)
    """
    
    dropout_prob: float = 0.1
    channel_dim: int = -1
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if np.random.random() < self.dropout_prob:
            channel_size = tensor.size(self.channel_dim)
            mask = torch.ones(channel_size, device=tensor.device)
            # Randomly select channels to zero out
            for i in range(channel_size):
                if np.random.random() < self.dropout_prob:
                    mask[i] = 0.0
                    
            # Expand mask to match tensor shape
            dims = [1] * tensor.dim()
            dims[self.channel_dim] = channel_size
            mask = mask.view(*dims)
            
            return tensor * mask
        return tensor


@dataclass
class RandomTimeReverse:
    """Randomly reverses the time dimension of the signal.
    
    This can help the model become invariant to the direction of movements.
    
    Args:
        prob (float): Probability of applying time reversal. (default: 0.5)
    """
    
    prob: float = 0.5
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if np.random.random() < self.prob:
            return torch.flip(tensor, [0])  # Flip the time dimension (dim 0)
        return tensor


@dataclass
class TimeWarping:
    """Applies random time warping to the EMG signal.
    
    This simulates variations in movement speed and timing during gesture execution.
    Uses piecewise linear interpolation to stretch or compress different segments
    of the signal.
    
    Args:
        num_control_points (int): Number of control points for warping.
            (default: 5)
        warp_strength (float): Maximum displacement strength as a ratio of 
            signal length. (default: 0.2)
    """
    
    num_control_points: int = 5
    warp_strength: float = 0.2
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        T = tensor.shape[0]  # Time dimension length
        
        if T <= self.num_control_points:
            return tensor  # Not enough points to warp
            
        # Create control points
        src_pts = np.linspace(0, T-1, self.num_control_points)
        
        # Create warped control points with random displacement
        max_displacement = int(T * self.warp_strength)
        displacements = np.random.randint(-max_displacement, max_displacement+1, size=self.num_control_points)
        # Keep endpoints fixed
        displacements[0] = 0
        displacements[-1] = 0
        
        dst_pts = src_pts + displacements
        # Ensure destination points remain in [0, T-1]
        dst_pts = np.clip(dst_pts, 0, T-1)
        dst_pts = np.unique(dst_pts)  # Remove duplicates if any
        
        if len(dst_pts) < 2:  # Need at least two points for interpolation
            return tensor
            
        # Build the mapping function based on these control points
        indices = torch.arange(T, device=tensor.device).float()
        # Convert points to PyTorch tensors
        src_pts_tensor = torch.tensor(src_pts, device=tensor.device).float()
        dst_pts_tensor = torch.tensor(dst_pts, device=tensor.device).float()
        
        # Perform piecewise linear interpolation
        warped_indices = torch.zeros(T, device=tensor.device)
        
        for i in range(len(dst_pts) - 1):
            mask = (indices >= src_pts_tensor[i]) & (indices <= src_pts_tensor[i+1])
            
            # Compute the local scaling factor
            src_span = src_pts_tensor[i+1] - src_pts_tensor[i]
            dst_span = dst_pts_tensor[i+1] - dst_pts_tensor[i]
            
            if src_span > 0:
                scale = dst_span / src_span
                # Apply linear interpolation to this segment
                warped_indices[mask] = dst_pts_tensor[i] + (indices[mask] - src_pts_tensor[i]) * scale
            
        # Round to nearest integer and clip to valid range
        warped_indices = torch.round(warped_indices).long().clamp(0, T-1)
        
        # Reindex the tensor
        warped_tensor = tensor[warped_indices]
        
        return warped_tensor


@dataclass
class RandomFrequencyFilter:
    """Applies random frequency filtering to the EMG signal.
    
    This simulates different electrode characteristics and filtering properties.
    Implemented as a simple FFT-based filter.
    
    Args:
        filter_type (str): Type of filter to apply: 'lowpass', 'highpass', 
            'bandpass', or 'random'. If 'random', a type is chosen randomly 
            for each call. (default: 'random')
        cutoff_low_range (tuple): Range of normalized low cutoff frequencies.
            (default: (0.05, 0.2))
        cutoff_high_range (tuple): Range of normalized high cutoff frequencies.
            (default: (0.5, 0.95))
    """
    
    filter_type: str = 'random'
    cutoff_low_range: tuple[float, float] = (0.05, 0.2) 
    cutoff_high_range: tuple[float, float] = (0.5, 0.95)
    
    def __post_init__(self) -> None:
        assert self.filter_type in ['lowpass', 'highpass', 'bandpass', 'random']
        
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # Move time to last dimension for FFT
        orig_shape = tensor.shape
        tensor_fft = tensor.movedim(0, -1)
        reshape_dims = tensor_fft.shape
        # Flatten all dimensions except the last one (time)
        tensor_fft = tensor_fft.reshape(-1, reshape_dims[-1])
        
        # Apply FFT
        fft_result = torch.fft.rfft(tensor_fft, dim=-1)
        
        # Create frequency mask
        freq_dim = fft_result.shape[-1]
        freq_mask = torch.ones(freq_dim, device=tensor.device)
        
        # Choose filter type
        if self.filter_type == 'random':
            chosen_filter = np.random.choice(['lowpass', 'highpass', 'bandpass'])
        else:
            chosen_filter = self.filter_type
        
        # Create the mask based on filter type
        if chosen_filter == 'lowpass':
            cutoff = int(np.random.uniform(self.cutoff_low_range[0], self.cutoff_high_range[1]) * freq_dim)
            freq_mask[cutoff:] = 0.0
        elif chosen_filter == 'highpass':
            cutoff = int(np.random.uniform(self.cutoff_low_range[0], self.cutoff_high_range[1]) * freq_dim)
            freq_mask[:cutoff] = 0.0
        elif chosen_filter == 'bandpass':
            low_cutoff = int(np.random.uniform(*self.cutoff_low_range) * freq_dim)
            high_cutoff = int(np.random.uniform(*self.cutoff_high_range) * freq_dim)
            if low_cutoff >= high_cutoff:
                low_cutoff, high_cutoff = high_cutoff, low_cutoff
            freq_mask[:low_cutoff] = 0.0
            freq_mask[high_cutoff:] = 0.0
        
        # Apply mask
        filtered_fft = fft_result * freq_mask
        
        # Inverse FFT
        filtered_signal = torch.fft.irfft(filtered_fft, n=reshape_dims[-1], dim=-1)
        
        # Reshape back to original dimensions
        filtered_signal = filtered_signal.reshape(reshape_dims)
        # Move time back to first dimension
        filtered_signal = filtered_signal.movedim(-1, 0)
        
        return filtered_signal


@dataclass
class MixUp:
    """Applies MixUp augmentation between two EMG signals.
    
    This randomly mixes two samples to create a new synthetic sample.
    MixUp is effective for improving generalization.
    
    Note: This transform should be applied at the batch level, not on individual samples.
    
    Args:
        alpha (float): Parameter for the beta distribution used to sample
            the mixing ratio. (default: 0.2)
        batch_dim (int): The batch dimension of the input tensor. (default: 1)
    """
    
    alpha: float = 0.2
    batch_dim: int = 1
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size = tensor.size(self.batch_dim)
        
        if batch_size < 2:
            return tensor  # Need at least 2 samples for mixup
            
        # Generate mixing weights from beta distribution
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
            
        # Ensure lam is not too extreme
        lam = max(lam, 1 - lam)
        
        # Generate random indices for mixing
        indices = torch.randperm(batch_size, device=tensor.device)
        
        # Get sample shape to properly broadcast lam
        shape = [1] * tensor.dim()
        shape[self.batch_dim] = batch_size
        lam_tensor = torch.ones(shape, device=tensor.device) * lam
        
        # Create mixed samples
        mixed = lam_tensor * tensor + (1 - lam_tensor) * tensor.index_select(self.batch_dim, indices)
        
        return mixed