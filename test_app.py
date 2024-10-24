import unittest
import json
import requests
import csv
import time
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from application import application


class TestFakeNewsDetector(unittest.TestCase):
    def setUp(self):
        """Initialize test environment before each test"""
        self.app = application.test_client()
        # Test cases with two fake and two real news examples
        self.test_cases = {
            'fake_news_1': "BREAKING: Scientists discover that chocolate cures all diseases overnight!",
            'fake_news_2': "Government confirms aliens are controlling all world leaders",
            'real_news_1': "Local city council approves new budget for infrastructure improvements",
            'real_news_2': "Study finds regular exercise helps reduce risk of heart disease"
        }

        # Create results directory if it doesn't exist
        self.results_dir = 'test_results'
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

        # Initialize CSV for detailed timing data
        self.csv_path = os.path.join(self.results_dir,
                                     f'api_timing_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(
                ['timestamp', 'test_case', 'iteration', 'request_time', 'response_time', 'total_time', 'prediction'])

    def test_api_performance(self):
        """Run performance tests for all test cases"""
        results = {}

        for case_name, text in self.test_cases.items():
            print(f"\nRunning performance test for: {case_name}")
            results[case_name] = []

            for i in range(100):  # 100 iterations per test case
                # Capture pre-request timestamp
                start_timestamp = datetime.now().isoformat()

                # Time the actual request
                request_start = time.time()
                response = self.app.post('/predict',
                                         json={'text': text},
                                         content_type='application/json')
                request_time = time.time() - request_start

                # Process response
                response_start = time.time()
                data = json.loads(response.data)
                response_time = time.time() - response_start

                # Calculate total time
                total_time = request_time + response_time
                results[case_name].append(total_time)

                # Save detailed timing data
                with open(self.csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        start_timestamp,
                        case_name,
                        i + 1,
                        request_time,
                        response_time,
                        total_time,
                        data['prediction']
                    ])

                if (i + 1) % 20 == 0:
                    print(f"Completed {i + 1} iterations...")

        self.generate_performance_report(results)

    def generate_performance_report(self, results):
        """Generate performance visualization and statistics"""
        # Create DataFrame for plotting
        df_data = []
        for case_name, latencies in results.items():
            for latency in latencies:
                df_data.append({'Test Case': case_name, 'Latency (s)': latency})
        df = pd.DataFrame(df_data)

        # Generate boxplot
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Test Case', y='Latency (s)', data=df)
        plt.title('API Latency Distribution by Test Case')
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(self.results_dir, 'latency_boxplot.png')
        plt.savefig(plot_path)
        plt.close()

        # Generate statistics report
        stats_path = os.path.join(self.results_dir, 'performance_stats.txt')
        with open(stats_path, 'w') as f:
            f.write("Performance Statistics\n")
            f.write("=====================\n\n")

            for case_name, latencies in results.items():
                f.write(f"\n{case_name}:\n")
                f.write(f"  Average Latency: {sum(latencies) / len(latencies):.4f} seconds\n")
                f.write(f"  Min Latency: {min(latencies):.4f} seconds\n")
                f.write(f"  Max Latency: {max(latencies):.4f} seconds\n")
                f.write(f"  95th Percentile: {sorted(latencies)[int(len(latencies) * 0.95)]:.4f} seconds\n")


def run_tests():
    """Run the complete test suite"""
    # Create test instance
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestFakeNewsDetector)
    unittest.TextTestRunner(verbosity=2).run(test_suite)


if __name__ == '__main__':
    run_tests()