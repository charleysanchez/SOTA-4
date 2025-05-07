# SOTA-4: EMG-to-QWERTY Neural Interface

**Winter 2025 - C147/247 Final Project**  
*Instructor: Professor Jonathan Kao*

This project builds upon Meta's [emg2qwerty](https://github.com/facebookresearch/emg2qwerty) to develop a neural interface translating electromyography (EMG) signals into QWERTY keyboard inputs. Our enhancements focus on preprocessing pipelines, model architecture, and evaluation metrics.

---

## 📂 Project Structure

- `config/`: Configuration files for experiments.
- `emg2qwerty/`: Core implementation from Meta's repository.
- `kao_data/`: Dataset used for training and evaluation.
- `meta_model/`: Custom model components and architectures.
- `preprocessing/`: Scripts for data cleaning and preprocessing.
- `scripts/`: Training, evaluation, and utility scripts.
- `notebooks/`: Jupyter notebooks for exploratory analysis.
- `tests/`: Unit tests to ensure code reliability.

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/charleysanchez/SOTA-4.git
cd SOTA-4
```

### 2. Create and Activate Environment

Using Conda:

```bash
conda env create -f environment.yml
conda activate sota4
```

Or using pip:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run Setup Script

```bash
python setup.py install
```

---

## 🚀 Usage

### Training the Model

```bash
python scripts/train_model.py --config config/train_config.yaml
```

### Evaluating the Model

```bash
python scripts/evaluate_model.py --config config/eval_config.yaml
```

---

## 📊 Results

Our model achieved a 15% improvement in accuracy over the baseline. Detailed results and analysis are available in the [C247_Final_Report.pdf](./C247_Final_Report.pdf).

---

## 🤝 Contributing

We welcome contributions! Please read our [Code of Conduct](./CODE_OF_CONDUCT.md) before contributing.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE.txt](./LICENSE.txt) file for details.

---

## 📬 Contact

For questions or feedback, please open an issue or contact [@charleysanchez](https://github.com/charleysanchez).
