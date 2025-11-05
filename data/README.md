# Data Transformer for VLM/LLM Training

## Quick Start

### 1. Basic Transformation

```bash
python data_transformer.py \
    -d input.json \
    -c config.yaml \
    -o output.json
```

### 2. Multiple Datasets with Report

```bash
python data_transformer.py \
    -d dataset1.json \
    -d dataset2.json \
    -d dataset3.json \
    -c config.yaml \
    -o training_data.json \
    --report
```

### 3. Custom Report Path

```bash
python data_transformer.py \
    -d data.json \
    -c config.yaml \
    -o output.json \
    --report analysis_report.pdf
```

## Input Data Format

Your input JSON should contain records with numeric or text fields:

```json
[
    {
        "path": "video1.mp4",
        "scores": {
            "MOS_minmax": 0.9104,
            "MOS_raw": 87.30
        }
    },
    {
        "path": "video2.mp4",
        "scores": {
            "MOS_minmax": 0.8169,
            "MOS_raw": 80.04
        }
    }
]
```

## Configuration File

Create a YAML or JSON config file to define transformations:

### Simple Example

```yaml
filters:
  - field: path
    condition: not_null

output_format:
  video_path:
    type: get_field
    field: path

  quality_label:
    type: map_field
    field: scores/MOS_minmax
    mapping:
      - min: 0.0
        max: 0.2
        label: bad
      - min: 0.2
        max: 0.4
        label: poor
      - min: 0.4
        max: 0.6
        label: fair
      - min: 0.6
        max: 0.8
        label: good
      - min: 0.8
        max: 1.0
        label: excellent
        inclusive_max: true
```

### VLM Training Format

```yaml
filters:
  - field: path
    condition: not_null

output_format:
  image:
    type: get_field
    field: path

  conversations:
    type: template
    template: '[{{"from": "human", "value": "What is the quality of this video?"}}, {{"from": "gpt", "value": "The video quality is {quality}. The MOS score is {score:.2f}."}}]'
    variables:
      quality:
        type: map_field
        field: scores/MOS_minmax
        mapping:
          - min: 0.0
            max: 0.2
            label: bad
          - min: 0.2
            max: 0.4
            label: poor
          - min: 0.4
            max: 0.6
            label: fair
          - min: 0.6
            max: 0.8
            label: good
          - min: 0.8
            max: 1.0
            label: excellent
            inclusive_max: true
      score:
        type: get_field
        field: scores/MOS_minmax
```

**Note**: When using JSON in templates, escape braces: `{{` and `}}` for literals, `{variable}` for placeholders.

## Configuration Options

### Filters

Filter records before processing:

```yaml
filters:
  - field: path
    condition: not_null  # Options: not_null, is_null
```

### Output Format Types

#### 1. `get_field` - Extract a field

```yaml
video_path:
  type: get_field
  field: path  # Supports nested: "scores/MOS_minmax"
```

#### 2. `map_field` - Map values to labels

```yaml
quality:
  type: map_field
  field: scores/MOS_minmax
  mapping:
    - min: 0.0
      max: 0.5
      label: low
    - min: 0.5
      max: 1.0
      label: high
      inclusive_max: true  # Include upper bound
```

#### 3. `template` - Format strings with variables

```yaml
response:
  type: template
  template: "Quality: {quality}, Score: {score:.2f}"
  variables:
    quality:
      type: map_field
      field: scores/MOS_minmax
      mapping: [...]
    score:
      type: get_field
      field: scores/MOS_minmax
```

## PDF Report Contents

When using `--report`, the generated PDF includes:

### 1. Title Page
- Generation timestamp
- Input file count
- Record counts

### 2. Combined Distribution Plots
- Histograms with KDE overlays
- All numeric fields from input data
- Mean, std deviation, sample size

### 3. Per-Dataset Comparisons
- Box plots comparing datasets
- Overlaid histogram distributions
- Useful for identifying biases

### 4. Label Distribution Charts
- Bar charts with percentages
- Shows final mapped labels
- Identifies class imbalance

### 5. Summary Statistics
- Dataset record counts
- Numeric field statistics (mean, min, max, median)
- Label frequency breakdown

## Command-Line Reference

```
usage: data_transformer.py [-h] -d FILE -c FILE -o FILE [--report [PDF_FILE]]

Transform JSON datasets into VLM/LLM training data with optional reporting

Arguments:
  -d, --data FILE       Input JSON file (can be repeated for multiple files)
  -c, --config FILE     Configuration file (YAML or JSON)
  -o, --output FILE     Output JSON file path
  --report [PDF_FILE]   Generate PDF report (optional: custom path)
  -h, --help            Show this help message
```

## Use Cases

### Video Quality Assessment
```bash
# Transform video quality scores to training data
python data_transformer.py \
    -d video_dataset.json \
    -c quality_config.yaml \
    -o vlm_training.json \
    --report quality_analysis.pdf
```

### Multi-Dataset Training
```bash
# Combine multiple datasets with analysis
python data_transformer.py \
    -d youtube.json \
    -d netflix.json \
    -d tiktok.json \
    -c config.yaml \
    -o combined_training.json \
    --report dataset_comparison.pdf
```

### Quality Assurance
```bash
# Generate report to check data before training
python data_transformer.py \
    -d raw_data.json \
    -c config.yaml \
    -o processed.json \
    --report qa_report.pdf
```

## Examples

See the `examples/` directory for complete examples:

- `config_simple.json` - Basic quality label mapping
- `config_vlm_format.json` - VLM conversation format
- `config_instruction_format.json` - Instruction-following format
- `config_annotated.yaml` - Fully documented config

## Tips & Best Practices

### Field Path Notation
Both `/` and `.` work as separators:
- `scores/MOS_minmax` ✅
- `scores.MOS_minmax` ✅

### Template Escaping
Use double braces for JSON in templates:
- `{{"key": "value"}}` → `{"key": "value"}`
- `{variable}` → replaced with actual value

### Label Mapping
- Use `inclusive_max: true` for the last interval to include 1.0
- Labels are case-sensitive
- Unmapped values get "out_of_range" label

### Performance
- Basic transformation: ~1s per 1,000 records
- With report: ~10s per 10,000 records
- Use `--report` only when needed for analysis

### Multiple Datasets
- Records are combined in order of `-d` arguments
- Per-dataset plots help identify distribution differences
- Useful for detecting dataset biases

## Troubleshooting

### KeyError: '"from"'
**Problem**: Template has unescaped JSON braces

**Solution**: Double the braces: `{{"from": "human"}}` instead of `{"from": "human"}`

### No plots in report
**Problem**: No numeric fields detected in input data

**Solution**: Ensure your JSON contains numeric fields (int/float types)

### Label distribution empty
**Problem**: No `map_field` transformations in config

**Solution**: Add at least one field with `type: map_field` in `output_format`

### Import errors
**Problem**: Missing dependencies

**Solution**: 
```bash
pip install pyyaml matplotlib seaborn scipy numpy
```

### Large datasets slow
**Problem**: Report generation takes too long

**Solution**: 
- Run without `--report` for production pipelines
- Use sampling for exploratory analysis
- Generate reports only for validation

## Advanced Usage

### Programmatic API

```python
from data_transformer import DataTransformer

# Initialize
transformer = DataTransformer('config.yaml')

# Process files
transformer.process_files(
    input_paths=['data1.json', 'data2.json'],
    output_path='output.json',
    generate_report=True,
    report_path='report.pdf'
)
```

### Custom Field Extraction

```python
# Access transformer methods directly
transformer = DataTransformer('config.yaml')

# Get nested field
value = transformer.get_field(item, 'scores/MOS_minmax')

# Map value to label
label = transformer.map_field(value, mapping_rules)
```

