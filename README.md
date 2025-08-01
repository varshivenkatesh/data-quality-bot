# dqbot: Data Quality Bot

`dqbot` is a CLI tool that automates data quality checks, supports multiple file formats, and even visualizes issues for better data transformation decisions.

## Usage

- Current version supports `.csv`, `.xlsx`, `.xls`, `.json` file formats
- CLI usage with `--file` or `--folder` to run analytics for data
- Add custom quality check metric configs in `.yaml` file, or use default config
- Automated visual reports with `--vis`
- Runs locally and supports data extraction from Azure container

---

## 📦 Installation

Install via pip:

```bash
pip install dqbot
```
Explore commands:
```bash
dqbot --help
