
import json
import yaml
from typing import Any, Dict, List, Union, Optional
from pathlib import Path
from collections import Counter
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import numpy as np
from datetime import datetime


class DataTransformer:
    """Transform JSON datasets into VLM/LLM training data using configuration."""

    def __init__(self, config_path: str):
        """Initialize transformer with configuration file."""
        self.config = self._load_config(config_path)

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML or JSON file."""
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                return yaml.safe_load(f)
            else:
                return json.load(f)

    def get_field(self, item: Dict, field_path: str) -> Any:
        """
        Extract a field from nested JSON using path notation.

        Args:
            item: Dictionary to extract from
            field_path: Path to field (e.g., "scores/MOS_minmax" or "scores.MOS_minmax")

        Returns:
            Field value or None if not found
        """
        # Support both / and . as separators
        parts = field_path.replace('/', '.').split('.')

        current = item
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current

    def map_field(self, value: float, mapping: List[Dict]) -> str:
        """
        Map a numeric value to text based on interval mappings.

        Args:
            value: Numeric value to map
            mapping: List of mapping rules with 'min', 'max', 'label', 'inclusive_max'

        Returns:
            Mapped text label
        """
        if value is None:
            return "unknown"

        for rule in mapping:
            min_val = rule.get('min', float('-inf'))
            max_val = rule.get('max', float('inf'))
            inclusive_max = rule.get('inclusive_max', False)

            if inclusive_max:
                if min_val <= value <= max_val:
                    return rule['label']
            else:
                if min_val <= value < max_val:
                    return rule['label']

        return "out_of_range"

    def format_template(self, item: Dict, template: str, variables: Dict[str, Any]) -> str:
        """
        Format a template string with variables extracted from the item.

        Args:
            item: Data item
            template: Template string with {variable} placeholders
            variables: Variable definitions from config

        Returns:
            Formatted string
        """
        values = {}

        for var_name, var_config in variables.items():
            if var_config['type'] == 'get_field':
                values[var_name] = self.get_field(item, var_config['field'])

            elif var_config['type'] == 'map_field':
                raw_value = self.get_field(item, var_config['field'])
                values[var_name] = self.map_field(raw_value, var_config['mapping'])

            elif var_config['type'] == 'constant':
                values[var_name] = var_config['value']

        return template.format(**values)

    def transform(self, input_data: List[Dict]) -> List[Dict]:
        """
        Transform input data according to configuration.

        Args:
            input_data: List of input data items

        Returns:
            List of transformed training examples
        """
        output = []

        for item in input_data:
            # Apply filters if specified
            if 'filters' in self.config:
                skip = False
                for filter_rule in self.config['filters']:
                    field_value = self.get_field(item, filter_rule['field'])

                    if filter_rule['condition'] == 'not_null':
                        if field_value is None:
                            skip = True
                            break
                    elif filter_rule['condition'] == 'is_null':
                        if field_value is not None:
                            skip = True
                            break

                if skip:
                    continue

            # Create output record
            output_record = {}

            for output_field, field_config in self.config['output_format'].items():
                if field_config['type'] == 'template':
                    output_record[output_field] = self.format_template(
                        item, 
                        field_config['template'],
                        field_config['variables']
                    )
                elif field_config['type'] == 'get_field':
                    output_record[output_field] = self.get_field(item, field_config['field'])
                elif field_config['type'] == 'map_field':
                    raw_value = self.get_field(item, field_config['field'])
                    output_record[output_field] = self.map_field(raw_value, field_config['mapping'])

            output.append(output_record)

        return output

    def _extract_numeric_fields(self, data: List[Dict]) -> Dict[str, List[float]]:
        """Extract all numeric fields from data for visualization."""
        numeric_data = {}

        # Try to find numeric fields by exploring the data structure
        if not data:
            return numeric_data

        # Get all possible field paths
        def get_all_paths(d, prefix=''):
            paths = []
            for key, value in d.items():
                path = f"{prefix}/{key}" if prefix else key
                if isinstance(value, dict):
                    paths.extend(get_all_paths(value, path))
                elif isinstance(value, (int, float)) and not isinstance(value, bool):
                    paths.append(path)
            return paths

        # Get field paths from first item
        if data and isinstance(data[0], dict):
            paths = get_all_paths(data[0])

            # Extract values for each path
            for path in paths:
                values = []
                for item in data:
                    val = self.get_field(item, path)
                    if val is not None and isinstance(val, (int, float)):
                        values.append(float(val))
                if values:
                    numeric_data[path] = values

        return numeric_data

    def _extract_mapped_labels(self, input_data: List[Dict]) -> Dict[str, List[str]]:
        """
        Extract mapped labels by applying map_field transformations to input data.
        This shows the final distribution of labels after mapping.
        """
        label_data = {}

        # Look through output_format for map_field configurations
        for field_name, field_config in self.config.get('output_format', {}).items():
            # Direct map_field
            if field_config.get('type') == 'map_field':
                labels = []
                for item in input_data:
                    raw_value = self.get_field(item, field_config['field'])
                    label = self.map_field(raw_value, field_config['mapping'])
                    labels.append(label)
                if labels:
                    label_data[f"{field_name}"] = labels

            # map_field inside template variables
            elif field_config.get('type') == 'template':
                variables = field_config.get('variables', {})
                for var_name, var_config in variables.items():
                    if var_config.get('type') == 'map_field':
                        labels = []
                        for item in input_data:
                            raw_value = self.get_field(item, var_config['field'])
                            label = self.map_field(raw_value, var_config['mapping'])
                            labels.append(label)
                        if labels:
                            label_data[f"{field_name}.{var_name}"] = labels

        return label_data

    def generate_report(self, input_files: List[str], output_data: List[Dict], 
                       per_dataset_data: Dict[str, List[Dict]], 
                       all_input_data: List[Dict], report_path: str):
        """
        Generate PDF report with data distribution visualizations.

        Args:
            input_files: List of input file paths
            output_data: Transformed output data
            per_dataset_data: Dictionary mapping filename to its data
            all_input_data: Combined input data (for label mapping)
            report_path: Path to save the PDF report
        """
        print(f"\nGenerating report: {report_path}")

        with PdfPages(report_path) as pdf:
            # Title page
            fig = plt.figure(figsize=(11, 8.5))
            fig.text(0.5, 0.6, 'Data Transformation Report', 
                    ha='center', va='center', fontsize=28, weight='bold')
            fig.text(0.5, 0.5, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                    ha='center', va='center', fontsize=14)
            fig.text(0.5, 0.4, f'Total input files: {len(input_files)}',
                    ha='center', va='center', fontsize=12)
            fig.text(0.5, 0.35, f'Total input records: {len(all_input_data)}',
                    ha='center', va='center', fontsize=12)
            fig.text(0.5, 0.3, f'Total output records: {len(output_data)}',
                    ha='center', va='center', fontsize=12)
            plt.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

            # Combined distributions of input data
            numeric_fields = self._extract_numeric_fields(all_input_data)

            if numeric_fields:
                self._plot_combined_distributions(pdf, numeric_fields, "Input Data")

            # Per-dataset distributions
            if len(per_dataset_data) > 1:
                self._plot_per_dataset_distributions(pdf, per_dataset_data)

            # Output label distributions (mapped from input)
            label_data = self._extract_mapped_labels(all_input_data)
            if label_data:
                self._plot_label_distributions(pdf, label_data, len(all_input_data))

            # Summary statistics
            self._plot_summary_statistics(pdf, numeric_fields, per_dataset_data, label_data)

        print(f"✅ Report saved: {report_path}")

    def _plot_combined_distributions(self, pdf: PdfPages, numeric_fields: Dict[str, List[float]], 
                                     title_suffix: str = ""):
        """Plot combined distributions of all numeric fields."""
        num_fields = len(numeric_fields)
        if num_fields == 0:
            return

        # Create subplots
        cols = min(2, num_fields)
        rows = (num_fields + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(11, 4*rows))
        full_title = f'Combined Data Distributions'
        if title_suffix:
            full_title += f' ({title_suffix})'
        fig.suptitle(full_title, fontsize=16, weight='bold')

        if num_fields == 1:
            axes = [axes]
        elif rows == 1:
            axes = list(axes) if isinstance(axes, np.ndarray) else [axes]
        else:
            axes = axes.flatten()

        for idx, (field_name, values) in enumerate(numeric_fields.items()):
            ax = axes[idx]

            # Histogram with KDE
            ax.hist(values, bins=50, alpha=0.7, color='steelblue', edgecolor='black', density=True)

            # Add KDE
            try:
                from scipy import stats
                kde = stats.gaussian_kde(values)
                x_range = np.linspace(min(values), max(values), 200)
                ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
            except:
                pass

            ax.set_xlabel('Value', fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.set_title(f'{field_name}\nn={len(values)}, μ={np.mean(values):.3f}, σ={np.std(values):.3f}',
                        fontsize=10)
            ax.grid(True, alpha=0.3)

        # Hide extra subplots
        for idx in range(num_fields, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    def _plot_per_dataset_distributions(self, pdf: PdfPages, per_dataset_data: Dict[str, List[Dict]]):
        """Plot per-dataset distributions for comparison."""
        # Extract numeric fields from each dataset
        dataset_fields = {}
        for dataset_name, data in per_dataset_data.items():
            dataset_fields[dataset_name] = self._extract_numeric_fields(data)

        # Find common fields across all datasets
        all_fields = set()
        for fields in dataset_fields.values():
            all_fields.update(fields.keys())

        # Plot each field across datasets
        for field_name in sorted(all_fields):
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            fig.suptitle(f'Per-Dataset Comparison: {field_name}', fontsize=14, weight='bold')

            # Box plot
            box_data = []
            labels = []
            for dataset_name, fields in dataset_fields.items():
                if field_name in fields:
                    box_data.append(fields[field_name])
                    labels.append(Path(dataset_name).stem[:20])  # Truncate long names

            if box_data:
                bp = axes[0].boxplot(box_data, labels=labels, patch_artist=True)
                for patch in bp['boxes']:
                    patch.set_facecolor('lightblue')
                axes[0].set_ylabel('Value', fontsize=10)
                axes[0].set_title('Box Plot Comparison', fontsize=11)
                axes[0].grid(True, alpha=0.3, axis='y')
                axes[0].tick_params(axis='x', rotation=45)

                # Overlaid histograms
                colors = plt.cm.tab10(np.linspace(0, 1, len(box_data)))
                for idx, (data, label) in enumerate(zip(box_data, labels)):
                    axes[1].hist(data, bins=30, alpha=0.5, label=f'{label} (n={len(data)})',
                               color=colors[idx], edgecolor='black')

                axes[1].set_xlabel('Value', fontsize=10)
                axes[1].set_ylabel('Frequency', fontsize=10)
                axes[1].set_title('Distribution Overlay', fontsize=11)
                axes[1].legend(fontsize=8, loc='best')
                axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

    def _plot_label_distributions(self, pdf: PdfPages, label_data: Dict[str, List[str]], 
                                  total_records: int):
        """Plot distribution of mapped labels."""
        num_label_fields = len(label_data)
        if num_label_fields == 0:
            return

        cols = min(2, num_label_fields)
        rows = (num_label_fields + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(11, 4*rows))
        fig.suptitle('Mapped Label Distributions (Final Output)', fontsize=16, weight='bold')

        if num_label_fields == 1:
            axes = [axes]
        elif rows == 1:
            axes = list(axes) if isinstance(axes, np.ndarray) else [axes]
        else:
            axes = axes.flatten()

        for idx, (field_name, labels) in enumerate(label_data.items()):
            ax = axes[idx]

            # Count labels
            label_counts = Counter(labels)
            sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)

            labels_list = [item[0] for item in sorted_labels]
            counts_list = [item[1] for item in sorted_labels]

            # Bar plot
            colors = plt.cm.Spectral(np.linspace(0.2, 0.8, len(labels_list)))
            bars = ax.bar(range(len(labels_list)), counts_list, color=colors, edgecolor='black', linewidth=1.5)
            ax.set_xticks(range(len(labels_list)))
            ax.set_xticklabels(labels_list, rotation=45, ha='right')
            ax.set_ylabel('Count', fontsize=10)
            ax.set_title(f'{field_name}\n(Total: {len(labels)} records)', fontsize=11)
            ax.grid(True, alpha=0.3, axis='y')

            # Add count labels on bars
            for bar, count in zip(bars, counts_list):
                height = bar.get_height()
                percentage = count / len(labels) * 100
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{count}\n({percentage:.1f}%)',
                       ha='center', va='bottom', fontsize=9, weight='bold')

        # Hide extra subplots
        for idx in range(num_label_fields, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    def _plot_summary_statistics(self, pdf: PdfPages, numeric_fields: Dict[str, List[float]],
                                 per_dataset_data: Dict[str, List[Dict]],
                                 label_data: Dict[str, List[str]]):
        """Plot summary statistics table."""
        fig, ax = plt.subplots(figsize=(11, 8.5))
        fig.suptitle('Summary Statistics', fontsize=16, weight='bold')
        ax.axis('off')

        y_position = 0.95

        # Dataset summary
        ax.text(0.1, y_position, 'Dataset Summary:', fontsize=14, weight='bold', transform=ax.transAxes)
        y_position -= 0.05

        for dataset_name, data in per_dataset_data.items():
            ax.text(0.15, y_position, f'• {Path(dataset_name).name}: {len(data)} records',
                   fontsize=11, transform=ax.transAxes)
            y_position -= 0.04

        y_position -= 0.05

        # Numeric field statistics
        if numeric_fields:
            ax.text(0.1, y_position, 'Input Data - Numeric Field Statistics:', fontsize=14, weight='bold',
                   transform=ax.transAxes)
            y_position -= 0.05

            for field_name, values in numeric_fields.items():
                stats_text = (f'• {field_name}:\n'
                            f'  Count: {len(values)}, Mean: {np.mean(values):.4f}, '
                            f'Std: {np.std(values):.4f}\n'
                            f'  Min: {np.min(values):.4f}, Max: {np.max(values):.4f}, '
                            f'Median: {np.median(values):.4f}')
                ax.text(0.15, y_position, stats_text, fontsize=10, transform=ax.transAxes,
                       verticalalignment='top')
                y_position -= 0.12
                if y_position < 0.1:
                    break

        # Label distribution summary
        if label_data and y_position > 0.2:
            y_position -= 0.03
            ax.text(0.1, y_position, 'Output Data - Label Distribution Summary:', fontsize=14, weight='bold',
                   transform=ax.transAxes)
            y_position -= 0.05

            for field_name, labels in label_data.items():
                label_counts = Counter(labels)
                ax.text(0.15, y_position, f'• {field_name}:', fontsize=11, weight='bold',
                       transform=ax.transAxes)
                y_position -= 0.04

                for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
                    percentage = count / len(labels) * 100
                    ax.text(0.20, y_position, f'{label}: {count} ({percentage:.1f}%)',
                           fontsize=10, transform=ax.transAxes)
                    y_position -= 0.03
                    if y_position < 0.05:
                        break

                y_position -= 0.02
                if y_position < 0.1:
                    break

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    def process_files(self, input_paths: List[str], output_path: str, generate_report: bool = False,
                     report_path: Optional[str] = None):
        """
        Process multiple JSON input files and save combined transformed output.

        Args:
            input_paths: List of paths to input JSON files
            output_path: Path to output JSON file
            generate_report: Whether to generate a PDF report
            report_path: Path to save the PDF report (default: output_path.replace('.json', '_report.pdf'))
        """
        per_dataset_data = {}
        all_input_data = []

        # Load all input files
        for input_path in input_paths:
            print(f"Loading: {input_path}")
            with open(input_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    per_dataset_data[input_path] = data
                    all_input_data.extend(data)
                else:
                    per_dataset_data[input_path] = [data]
                    all_input_data.append(data)

        print(f"\nTotal input records: {len(all_input_data)}")

        # Transform
        output_data = self.transform(all_input_data)

        # Save output
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"Generated output records: {len(output_data)}")
        print(f"Output saved to: {output_path}")

        # Generate report if requested
        if generate_report:
            if report_path is None:
                report_path = output_path.replace('.json', '_report.pdf')
                if report_path == output_path:  # If no .json extension
                    report_path = output_path + '_report.pdf'

            self.generate_report(input_paths, output_data, per_dataset_data, all_input_data, report_path)


def main():
    """Main entry point for the script."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Transform JSON datasets into VLM/LLM training data with optional reporting',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic transformation
  python data_transformer.py -d input.json -c config.yaml -o output.json

  # Multiple input files with report
  python data_transformer.py -d data1.json -d data2.json -c config.yaml -o output.json --report

  # Custom report path
  python data_transformer.py -d data.json -c config.yaml -o output.json --report my_report.pdf
        """
    )

    parser.add_argument(
        '-d', '--data',
        action='append',
        required=True,
        dest='input_files',
        metavar='FILE',
        help='Input JSON file path (can be specified multiple times for multiple files)'
    )

    parser.add_argument(
        '-c', '--config',
        required=True,
        dest='config_file',
        metavar='FILE',
        help='Configuration file path (YAML or JSON)'
    )

    parser.add_argument(
        '-o', '--output',
        required=True,
        dest='output_file',
        metavar='FILE',
        help='Output JSON file path'
    )

    parser.add_argument(
        '--report',
        nargs='?',
        const=True,
        default=False,
        dest='report',
        metavar='PDF_FILE',
        help='Generate PDF report with data distribution visualizations (optional: specify custom path)'
    )

    args = parser.parse_args()

    # Validate input files exist
    for input_file in args.input_files:
        if not Path(input_file).exists():
            parser.error(f"Input file not found: {input_file}")

    # Validate config file exists
    if not Path(args.config_file).exists():
        parser.error(f"Config file not found: {args.config_file}")

    # Determine report path
    report_path = None
    generate_report = False

    if args.report:
        generate_report = True
        if isinstance(args.report, str):
            report_path = args.report
        else:
            report_path = args.output_file.replace('.json', '_report.pdf')
            if report_path == args.output_file:
                report_path = args.output_file + '_report.pdf'

    # Process files
    transformer = DataTransformer(args.config_file)
    transformer.process_files(args.input_files, args.output_file, generate_report, report_path)


if __name__ == '__main__':
    main()
