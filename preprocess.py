import os
import xml.etree.ElementTree as ET
import pandas as pd
from typing import List, Dict
from tqdm import tqdm
import logging

class MedQuADPreprocessor:
    def __init__(self, input_dir: str, output_dir: str):
        """
        Initialize the MedQuAD preprocessor
        
        Args:
            input_dir (str): Path to MedQuAD-master directory
            output_dir (str): Path to save processed files
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('medquad_preprocessing.log'),
                logging.StreamHandler()
            ]
        )
        
    def parse_xml_file(self, file_path: str) -> List[Dict]:
        """
        Parse a single XML file and extract QA pairs
        
        Args:
            file_path (str): Path to XML file
            
        Returns:
            List[Dict]: List of dictionaries containing QA pairs
        """
        qa_pairs = []
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Extract focus/topic
            focus = root.find('Focus')
            focus_text = focus.text if focus is not None else "No Focus"
            
            # Get source information
            source = os.path.basename(file_path).replace('.xml', '')
            
            # Extract all QA pairs
            for qa_pair in root.findall('.//QAPair'):
                question = qa_pair.find('Question')
                answer = qa_pair.find('Answer')
                
                if question is not None and answer is not None:
                    qa_pairs.append({
                        'focus': focus_text,
                        'source': source,
                        'question': question.text or "No Question",
                        'answer': answer.text or "No Answer"
                    })
                    
        except ET.ParseError as e:
            logging.error(f"XML parsing error in {file_path}: {e}")
        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}")
            
        return qa_pairs

    def process_directory(self) -> pd.DataFrame:
        """
        Process all XML files in the MedQuAD directory structure
        
        Returns:
            pd.DataFrame: DataFrame containing all QA pairs
        """
        all_qa_pairs = []
        xml_files = []

        # First, collect all XML files
        for root, _, files in os.walk(self.input_dir):
            for file in files:
                if file.endswith('.xml'):
                    xml_files.append(os.path.join(root, file))

        # Process each XML file with progress bar
        for file_path in tqdm(xml_files, desc="Processing XML files"):
            qa_pairs = self.parse_xml_file(file_path)
            all_qa_pairs.extend(qa_pairs)
            
        # Convert to DataFrame
        df = pd.DataFrame(all_qa_pairs)
        return df

    def save_processed_data(self, df: pd.DataFrame):
        """
        Save processed data in multiple formats
        
        Args:
            df (pd.DataFrame): Processed DataFrame
        """
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save as CSV
        csv_path = os.path.join(self.output_dir, 'medical_qa.csv')
        df.to_csv(csv_path, index=False)
        logging.info(f"Saved CSV file to {csv_path}")
        
        # Save as Parquet (more efficient for large datasets)
        parquet_path = os.path.join(self.output_dir, 'medical_qa.parquet')
        df.to_parquet(parquet_path, index=False)
        logging.info(f"Saved Parquet file to {parquet_path}")
        
        # Generate and save dataset statistics
        stats = {
            'total_qa_pairs': len(df),
            'unique_topics': df['focus'].nunique(),
            'avg_question_length': df['question'].str.len().mean(),
            'avg_answer_length': df['answer'].str.len().mean()
        }
        
        stats_path = os.path.join(self.output_dir, 'dataset_stats.txt')
        with open(stats_path, 'w') as f:
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
        logging.info(f"Saved dataset statistics to {stats_path}")

def main():
    """Main function to run the preprocessing"""
    # Configure paths
    input_dir = "./MedQuAD-master"  # Path to MedQuAD dataset
    output_dir = "./processed_data"  # Path to save processed files
    
    # Initialize preprocessor
    preprocessor = MedQuADPreprocessor(input_dir, output_dir)
    
    try:
        # Process the dataset
        logging.info("Starting dataset preprocessing...")
        df = preprocessor.process_directory()
        
        # Save processed data
        preprocessor.save_processed_data(df)
        
        logging.info("Preprocessing completed successfully!")
        
        # Print basic statistics
        print("\nDataset Statistics:")
        print(f"Total QA pairs: {len(df)}")
        print(f"Unique topics: {df['focus'].nunique()}")
        print(f"Average question length: {df['question'].str.len().mean():.2f} characters")
        print(f"Average answer length: {df['answer'].str.len().mean():.2f} characters")
        
    except Exception as e:
        logging.error(f"Preprocessing failed: {e}")
        raise

if __name__ == "__main__":
    main()