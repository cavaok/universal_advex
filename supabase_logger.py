import os
from supabase import create_client
import torch
import numpy as np


class SupabaseLogger:
    def __init__(self):
        # Direct string assignment of credentials
        self.supabase_url = "https://fxwzblzdvwowourssdih.supabase.co"
        self.supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZ4d3pibHpkdndvd291cnNzZGloIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzY0NjA1NTUsImV4cCI6MjA1MjAzNjU1NX0.Kcc9aJmOcgn6xF76aqfGUs6rO9awnabimX8HJnPhzrQ"

        print(f"Initializing Supabase with URL: {self.supabase_url}")
        try:
            self.supabase = create_client(self.supabase_url, self.supabase_key)
            print("Supabase client created successfully")
        except Exception as e:
            print(f"Error creating Supabase client: {str(e)}")
            raise

    def _tensor_to_list(self, tensor):
        """Convert a PyTorch tensor to a list of values"""
        try:
            return tensor.cpu().detach().numpy().flatten().tolist()
        except Exception as e:
            print(f"Error in _tensor_to_list: {str(e)}")
            raise

    def log_result(self, case_idx, model_name, image, label, results):
        try:
            print(f"\nAttempting to log results for case {case_idx}, model {model_name}")

            data = {
                "case_idx": case_idx,
                "model_name": model_name,
                "image": self._tensor_to_list(image),
                "label": label.item() if torch.is_tensor(label) else label,
                "original_prediction": self._tensor_to_list(results["original_prediction"]),
                "adversarial_image": self._tensor_to_list(results["adversarial_image"]),
                "prediction": self._tensor_to_list(results["prediction"]),
                "label_kld": results["label_kld"],
                "mse": results["mse"],
                "frob": results["frob"]
            }

            print("Data prepared successfully")
            print(f"Attempting to insert into table 'adversarial_examples'")

            response = self.supabase.table("adversarial_examples").insert(data).execute()
            print(f"Insert response: {response}")
            return True

        except Exception as e:
            print(f"Error logging to Supabase: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
            return False

    @staticmethod
    def retrieve_image(pixel_array, shape=(28, 28)):
        """Convert stored pixel array back to proper image shape"""
        return np.array(pixel_array).reshape(shape)

    @staticmethod
    def retrieve_prediction(pred_array):
        """Convert stored prediction array back to proper shape"""
        return np.array(pred_array)  # Already in correct shape (length 10)