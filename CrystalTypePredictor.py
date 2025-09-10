import pandas as pd
import numpy as np
import joblib
import os


class CrystalTypeGridPredictor:
    """
    Generates a custom 2D grid of predicted crystal types using specified Na:Bi and F:Bi ratios.

    Usage:
        predictor = CrystalTypeGridPredictor("random_forest_gridsearch")
        na_list = [0.125, 0.25, ...]  # Your Na:Bi ratios
        f_list = [0.125, 0.25, ...]    # Your F:Bi ratios
        grid_df = predictor.generate_custom_grid(na_list, f_list, "custom_crystal_grid.csv")
    """

    def __init__(self, model_dir="data_20"):
        """
        Initialize with trained model artifacts.

        Args:
            model_dir (str): Directory containing trained model artifacts
        """
        self.model_dir = model_dir
        self._load_artifacts()

    def _load_artifacts(self):
        """Load trained model and scaler (no feature engineering artifacts)."""
        self.model = joblib.load(os.path.join(self.model_dir, 'optimized_model.pkl'))
        self.scaler = joblib.load(os.path.join(self.model_dir, 'scaler.pkl'))
        print("Successfully loaded prediction artifacts (no feature engineering)")

    def generate_custom_grid(self, na_list, f_list, output_path=None):
        """
        Create a custom 2D grid of predictions using specified Na:Bi and F:Bi ratios.

        Args:
            na_list (list): List of Na:Bi values
            f_list (list): List of F:Bi values
            output_path (str): Optional path to save CSV file

        Returns:
            pd.DataFrame: Grid of predicted crystal types with Na:Bi as index and F:Bi as columns
        """
        # Create all combinations of Na:Bi and F:Bi
        na_values, f_values = np.meshgrid(na_list, f_list)
        grid_points = np.vstack([na_values.ravel(), f_values.ravel()]).T

        # Create DataFrame with only the two original features
        input_df = pd.DataFrame(grid_points, columns=['Na:Bi', 'F:Bi'])

        # Apply normalization (no feature engineering)
        scaled_data = self.scaler.transform(input_df)

        # Make predictions
        predictions = self.model.predict(scaled_data)

        # Reshape to 2D grid (rows = Na:Bi, columns = F:Bi)
        prediction_grid = predictions.reshape(len(f_list), len(na_list)).T

        # Create DataFrame with Na:Bi as index and F:Bi as columns
        grid_df = pd.DataFrame(
            prediction_grid,
            index=na_list,
            columns=f_list
        )

        # Name index and columns
        grid_df.index.name = 'Na:Bi'
        grid_df.columns.name = 'F:Bi'

        # Save to CSV if requested
        if output_path:
            grid_df.to_csv(output_path)
            print(f"Custom prediction grid saved to {output_path}")

        return grid_df


# Your specific ratio lists
NA_BI = [
    0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1, 1.125, 1.025, 1.375, 1.5, 1.625, 1.75,
    1.875, 2, 2.2, 2.4, 2.6, 2.8, 3, 3.2, 3.4, 3.6, 3.8, 4, 4.25, 4.5, 4.75, 5, 5.25, 5.5,
    5.75, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5,
    15, 15.5, 16, 16.5, 17, 17.5, 18, 18.5, 19, 19.5, 20, 20.5, 21, 21.5, 22, 22.5, 23, 23.5, 24
]

F_BI = [
    0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.025, 1.375, 1.5, 1.625, 1.75,
    1.875, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.25, 4.5, 4.75, 5.0, 5.25,
    5.5, 5.75, 6.0, 6.5, 7.0, 7.5, 8.0, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
    23, 24, 25, 26, 27, 28, 29, 30, 31, 32
]

if __name__ == "__main__":
    # Initialize predictor with your model directory
    predictor = CrystalTypeGridPredictor("data_20")

    # Generate custom grid using your specified ratios
    grid_df = predictor.generate_custom_grid(
        na_list=NA_BI,
        f_list=F_BI,
        output_path="data_20/custom_grid.csv"
    )

    # Print a sample of the results
    print("\nSample of prediction grid:")
    print(grid_df.head(5).iloc[:, :5])  # Show first 5 rows and 5 columns

    # Print the dimensions
    print(f"\nGrid dimensions: {len(NA_BI)} Na:Bi ratios × {len(F_BI)} F:Bi ratios")