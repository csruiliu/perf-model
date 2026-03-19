import re
import os
import pandas as pd

from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter


class JobProcessor:
    """Handles metrics file processing"""
    GPU_PATTERN = re.compile(r'^GPU \d+\s')
    HEADER_PATTERN = re.compile(r'^#Entity')

    def __init__(self, num_gpu: int, metric_names: List[str]):
        self.num_gpu = num_gpu
        self.metric_names = metric_names

    @staticmethod
    def is_float(value: str) -> bool:
        """Check if a string can be converted to float"""
        try:
            float(value)
            return True
        except ValueError:
            return False

    @staticmethod
    def _count_zero(profiled_df: pd.DataFrame):
        # Filter for gract > 0.9
        filtered = profiled_df[profiled_df['GRACT'] > 0.9]
        total_samples = len(filtered)
        
        # Count zeros for each metric
        tensor_zeros = ((filtered['GRACT'] > 0.9) & (filtered['TENSO'] < 0.01)).sum()
        drama_zeros = ((filtered['GRACT'] > 0.9) & (filtered['DRAMA'] < 0.01)).sum()
        fp64_zeros = ((filtered['GRACT'] >= 0.9) & (filtered['FP64A'] < 0.01)).sum()
        fp32_zeros = ((filtered['GRACT'] >= 0.9) & (filtered['FP32A'] < 0.01)).sum()
        fp16_zeros = ((filtered['GRACT'] >= 0.9) & (filtered['FP16A'] < 0.01)).sum()
        print(f"Total Samples: {total_samples}, DRAMA Zero Samples: {drama_zeros}, TENSO Zero Samples: {tensor_zeros}, "
              f"FP64A Zero Samples: {fp64_zeros}, FP32A Zero Samples: {fp32_zeros}, FP16A Zero Samples: {fp16_zeros}")


    def _organize_by_file_content(self, all_files: List[Path]) -> List[str]:
        """Organize files by analyzing their content"""
        file_info = []
        
        for file_path in all_files:
            try:
                with open(file_path, 'r') as f:
                    content = ""
                    for i, line in enumerate(f):
                        content += line
                        if i > 10:
                            break
                
                gpu_pattern = re.compile(r'GPU (\d+)')
                gpu_matches = gpu_pattern.findall(content)
                
                if gpu_matches:
                    gpu_counter = Counter(gpu_matches)
                    most_common_gpu_id = int(gpu_counter.most_common(1)[0][0])
                    total_lines = len(gpu_matches)
                    file_info.append((file_path, most_common_gpu_id, total_lines))
                else:
                    print(f"Warning: No GPU data found in {file_path}")
                    file_info.append((file_path, -1, 0))
            
            except Exception as e:
                print(f"Warning: Could not read file {file_path}: {e}")
                file_info.append((file_path, -1, 0))
        
        # Sort by filename for deterministic ordering
        file_info.sort(key=lambda x: x[0].name)
        
        if len(file_info) != self.num_gpu:
            print(f"Content-based matching found {len(file_info)} valid files, expected {self.num_gpu}.")
            raise ValueError(f"Expected {self.num_gpu} files but found {len(file_info)}")
        
        # Sort by GPU ID and line count
        file_info.sort(key=lambda x: (x[1], x[2]))
        
        organized_files = [str(info[0]) for info in file_info]
        
        print("File organization by content analysis:")
        for logical_gpu_id, (file_path, detected_gpu_id, line_count) in enumerate(file_info):
            if detected_gpu_id >= 0:
                print(f"  Logical GPU {logical_gpu_id}: {file_path.name} "
                      f"(detected GPU {detected_gpu_id}, {line_count} data lines)")
            else:
                print(f"  Logical GPU {logical_gpu_id}: {file_path.name} "
                      f"(GPU ID unknown, {line_count} data lines)")
        
        return organized_files


    def _scan_and_organize_gpu_files(self, folder_path: str) -> List[str]:
        """Scan a folder for GPU data files and organize them by logical GPU ID"""
        folder_path = Path(folder_path)
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        file_patterns = ['*.out', '*.txt']
        all_files = []
        for pattern in file_patterns:
            all_files.extend(folder_path.glob(pattern))
        
        if not all_files:
            raise FileNotFoundError(f"No data files found in {folder_path}")
        
        print(f"Found {len(all_files)} potential GPU data files in {folder_path}")
        
        return self._organize_by_file_content(all_files)
    

    def process_files(self, dcgm_input: str) -> List[pd.DataFrame]:
        """Process input files or directory"""
        if os.path.isdir(dcgm_input):
            print(f"Processing folder: {dcgm_input}")
            file_paths = self._scan_and_organize_gpu_files(dcgm_input)
            return self._process_multiple_files(file_paths)
        elif os.path.isfile(dcgm_input):
            print(f"Processing single file with {self.num_gpu} GPUs: {dcgm_input}")
            return self._process_single_file(dcgm_input)
        else:
            raise ValueError(f"Input path '{dcgm_input}' is neither a valid file nor a directory")


    def _process_multiple_files(self, file_paths: List[str]) -> List[pd.DataFrame]:
        """Process multiple files, each containing single GPU data"""
        profiled_data = list()
        
        for logical_gpu_id, file_path in enumerate(file_paths):
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            print(f"Processing file {file_path} as logical GPU {logical_gpu_id}")
            
            with open(file_path, 'r') as file:
                lines = file.readlines()
            
            header_columns, metric_indices = self._parse_header(lines)
            gpu_data = self._extract_single_gpu_data(lines, metric_indices, len(header_columns))
            
            if gpu_data:
                df = pd.DataFrame(gpu_data, columns=self.metric_names)
                profiled_data.append(df)
                print(f"Logical GPU {logical_gpu_id}: Created DataFrame with {len(gpu_data)} rows")
            else:
                profiled_data.append(pd.DataFrame(columns=self.metric_names))
                print(f"Warning: No data found for logical GPU {logical_gpu_id} in file {file_path}")
        
        return profiled_data
    

    def _process_single_file(self, file_path: str) -> List[pd.DataFrame]:
        """Process a single file containing multiple GPU data"""
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        header_columns, metric_indices = self._parse_header(lines)
        gpu_data = self._extract_gpu_data(lines, metric_indices, len(header_columns))

        # only one gpu so just fetch the first item
        profiled_data = self._create_dataframes(gpu_data)[0]
        
        # count number of lines with (nearly) "zero" activities
        self._count_zero(profiled_data)

        return profiled_data


    def _extract_gpu_data(self, lines: List[str], metric_indices: List[int], 
                                num_columns: int) -> Dict[int, List[List[float]]]:
        """Extract data from a file with multiple GPUs"""
        gpu_data = {}
        
        for line in lines:
            if self.HEADER_PATTERN.match(line):
                continue
            if self.GPU_PATTERN.match(line):
                parts = re.split(r'\s{3,}', line.strip())
                
                # Extract GPU ID
                gpu_match = re.search(r'GPU (\d+)', parts[0])
                if not gpu_match:
                    continue
                
                gpu_id = int(gpu_match.group(1))
                
                # Extract numeric values
                values = parts[1:]
                # Creates a list of numeric values, converting 'n/a' strings to 0.0 and other valid numbers to floats
                numeric_values = [0.0 if v.strip().lower() == 'n/a' else float(v) 
                                  for v in values if self.is_float(v) or v.strip().lower() == 'n/a']
                
                if len(numeric_values) >= num_columns - 1:
                    selected_values = [numeric_values[i] for i in metric_indices]
                    
                    if gpu_id not in gpu_data:
                        gpu_data[gpu_id] = []
                    gpu_data[gpu_id].append(selected_values)
                else:
                    print(f"Warning: Line has insufficient data columns: {line.strip()}")
        
        return gpu_data


    def _create_dataframes(self, gpu_data: List[List[float]]) -> List[pd.DataFrame]:
        """Create DataFrames from GPU data dictionary"""
        gpu_dfs = []
        for gpu_id in sorted(gpu_data.keys()):
            if gpu_data[gpu_id]:
                df = pd.DataFrame(gpu_data[gpu_id], columns=self.metric_names)
                gpu_dfs.append(df)
            else:
                gpu_dfs.append(pd.DataFrame(columns=self.metric_names))
        
        return gpu_dfs


    def _parse_header(self, lines: List[str]) -> Tuple[List[str], List[int]]:
        """Parse header and find metric column indices"""
        for line in lines:
            if self.HEADER_PATTERN.match(line):
                header_columns = [col.strip() for col in re.split(r'\s{2,}', line.strip())]
                metric_indices = self._get_metric_indices(header_columns)
                return header_columns, metric_indices
        raise ValueError("Could not find header line in the data file")


    def _get_metric_indices(self, header_columns: List[str]) -> List[int]:
        """Map requested metrics to their column indices"""
        metric_indices = []
        for metric in self.metric_names:
            if metric not in header_columns:
                raise ValueError(
                    f"Metric '{metric}' not found in data file. "
                    f"Available metrics: {header_columns[1:]}"
                )
            metric_indices.append(header_columns.index(metric) - 1)
        return metric_indices