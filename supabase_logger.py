import os
from supabase import create_client
import torch
import numpy as np


class SupabaseLogger:
    def __init__(self):
        # Get these values from your Supabase project settings
        supabase_url = os.getenv("https://fxwzblzdvwowourssdih.supabase.coL")
        supabase_key = os.getenv("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZ4d3pibHpkdndvd291cnNzZGloIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzY0NjA1NTUsImV4cCI6MjA1MjAzNjU1NX0.Kcc9aJmOcgn6xF76aqfGUs6rO9awnabimX8HJnPhzrQ")

        if not supabase_url or not supabase_key:
            raise ValueError("Please set SUPABASE_URL and SUPABASE_KEY environment variables")

        self.supabase = create_client(supabase_url, supabase_key)

    def _tensor_to_list(self, tensor):
        """Convert a PyTorch tensor to a list of values"""
        return tensor.cpu().detach().numpy().flatten().tolist()

    def log_result(self, case_idx, model_name, image, label, results):
        try:
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

            self.supabase.table("adversarial_examples").insert(data).execute()
            return True

        except Exception as e:
            print(f"Error logging to Supabase: {str(e)}")
            return False

    def retrieve_image(self, pixel_array, shape=(28, 28)):
        """Convert stored pixel array back to proper image shape"""
        return np.array(pixel_array).reshape(shape)

    def retrieve_prediction(self, pred_array):
        """Convert stored prediction array back to proper shape"""
        return np.array(pred_array)  # Already in correct shape (length 10)