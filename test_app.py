import unittest
import json
import csv
import time
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from application import app


class TestFakeNewsDetector(unittest.TestCase):
    def setUp(self):
        """Initialize test environment before each test"""
        self.app = app.test_client()
        # Test cases with two fake and two real news examples
        self.test_cases = {
            'fake_news_1': "BREAKING: Scientists discover that chocolate cures all diseases overnight!",
            'fake_news_2': "Government confirms aliens are controlling all world leaders",
            'real_news_1': "Local city council approves new budget for infrastructure improvements",
            'real_news_2': "Study finds regular exercise helps reduce risk of heart disease"
        }

        # Create test_results directory if it doesn't exist
        if not os.path.exists('test_results'):
            os.makedirs('test_results')

    def test_prediction_endpoint(self):
        """Test basic functionality of the prediction endpoint"""
        print("\nFunctional Test Results:")
        print("-" * 50)

        for case_name, text in self.test_cases.items():
            response = self.app.post('/predict',
                                     json={'text': text},
                                     content_type='application/json')
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)

            print(f"\nTest Case: {case_name}")
            print(f"Text: {text}")
            print(f"Prediction: {'Fake' if data['prediction'] == 1 else 'Real'} News")
            print(f"Latency: {data['latency']:.4f} seconds")

    def perform_latency_test(self):
        """Perform latency testing with 100 API calls per test case"""
        if not hasattr(self, 'test_cases'):
            self.setUp()

        results = {}
        csv_path = os.path.join('test_results', 'latency_results.csv')

        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['test_case', 'iteration', 'latency', 'prediction']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for case_name, text in self.test_cases.items():
                results[case_name] = []
                print(f"\nTesting {case_name}...")

                for i in range(100):
                    start_time = time.time()
                    response = self.app.post('/predict',
                                             json={'text': text},
                                             content_type='application/json')
                    latency = time.time() - start_time
                    results[case_name].append(latency)

                    data = json.loads(response.data)
                    writer.writerow({
                        'test_case': case_name,
                        'iteration': i + 1,
                        'latency': latency,
                        'prediction': data['prediction']
                    })

                    if (i + 1) % 20 == 0:
                        print(f"Completed {i + 1} iterations...")

        return results

    def generate_latency_plots(self, results):
        """Generate boxplots for latency results"""
        df_data = []
        for case_name, latencies in results.items():
            for latency in latencies:
                df_data.append({'Test Case': case_name, 'Latency (s)': latency})
        df = pd.DataFrame(df_data)

        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Test Case', y='Latency (s)', data=df)
        plt.title('API Latency Distribution by Test Case')
        plt.xticks(rotation=45)
        plt.tight_layout()

        plot_path = os.path.join('test_results', 'latency_boxplot.png')
        plt.savefig(plot_path)
        plt.close()  # Close the figure to free memory

        print("\nLatency Statistics:")
        print("-" * 50)
        for case_name in results:
            latencies = results[case_name]
            avg_latency = sum(latencies) / len(latencies)
            max_latency = max(latencies)
            min_latency = min(latencies)
            print(f"\n{case_name}:")
            print(f"  Average: {avg_latency:.4f} seconds")
            print(f"  Maximum: {max_latency:.4f} seconds")
            print(f"  Minimum: {min_latency:.4f} seconds")


def run_all_tests():
    """Run all tests including latency tests and generate plots"""
    # Run unit tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFakeNewsDetector)
    unittest.TextTestRunner(verbosity=2).run(suite)

    # Run latency tests and generate plots
    tester = TestFakeNewsDetector()
    print("\nRunning latency tests...")
    print("=" * 50)
    results = tester.perform_latency_test()
    tester.generate_latency_plots(results)


if __name__ == '__main__':
    run_all_tests()