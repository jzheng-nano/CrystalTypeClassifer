import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (confusion_matrix, classification_report,
                             f1_score, precision_score, recall_score, accuracy_score,
                             roc_auc_score, ConfusionMatrixDisplay, roc_curve, auc)
from sklearn.preprocessing import label_binarize, StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
import os
import warnings
from itertools import cycle
import joblib
import time
import matplotlib as mpl
import copy
from scipy import interp


class CrystalTypeClassifier:
    """
    A machine learning pipeline for classifying crystal types based on chemical ratios.

    Attributes:
        output_dir (str): Directory to save all outputs
        RANDOM_STATE (int): Random seed for reproducibility
        data (pd.DataFrame): Raw input data
        X_train, X_test, y_train, y_test: Split datasets
        best_model: Optimized RandomForest classifier
        scaler: Fitted StandardScaler
        poly: Fitted PolynomialFeatures transformer
    """

    def __init__(self, data_path='data_20.csv', output_dir="data_20", random_state=42):
        """
        Initialize the classifier with configuration parameters.

        Args:
            data_path (str): Path to input CSV file
            output_dir (str): Directory to save outputs
            random_state (int): Random seed for reproducibility
        """
        self.output_dir = output_dir
        self.RANDOM_STATE = random_state
        os.makedirs(self.output_dir, exist_ok=True)
        warnings.filterwarnings("ignore")

        # Configure plotting aesthetics
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.size'] = 6.5
        plt.rcParams['axes.labelsize'] = 6.5
        plt.rcParams['xtick.labelsize'] = 7
        plt.rcParams['ytick.labelsize'] = 7

        # Load and prepare data
        self.data = pd.read_csv(data_path)
        print(f"Loaded dataset with {len(self.data)} samples and {self.data.shape[1] - 1} features")
        self._prepare_data()

        # Address class imbalance concerns
        class_counts = self.data['Crystal_Type'].value_counts()
        print("\nClass distribution:")
        for crystal_type, count in class_counts.items():
            print(f"Class {crystal_type}: {count} samples ({count / len(self.data):.1%})")

    def _prepare_data(self):
        """Load and split data into training and testing sets."""
        print("Using features: 'Na:Bi' (Sodium-to-Bismuth ratio), 'F:Bi' (Fluorine-to-Bismuth ratio)")
        print("Predicting target: 'Crystal_Type' (0=Others, 1=NaBiF4, 2=BiF3)")

        X = self.data[['Na:Bi', 'F:Bi']]
        y = self.data['Crystal_Type']

        # Stratified train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=self.RANDOM_STATE
        )

        # Reset indices for consistency
        self.X_train = self.X_train.reset_index(drop=True)
        self.X_test = self.X_test.reset_index(drop=True)
        self.y_train = self.y_train.reset_index(drop=True)
        self.y_test = self.y_test.reset_index(drop=True)

        print(f"Train size: {len(self.X_train)}, Test size: {len(self.X_test)}")

    def _add_features(self, df):
        """
        Engineer new features from base chemical ratios.

        Args:
            df (pd.DataFrame): Input dataframe with 'Na:Bi' and 'F:Bi' columns

        Returns:
            pd.DataFrame: Dataframe with additional engineered features
        """
        df = df.copy()
        df['Na:Bi/F:Bi'] = df['Na:Bi'] / (df['F:Bi'] + 1e-6)  # Avoid division by zero
        df['Na:Bi+F:Bi'] = df['Na:Bi'] + df['F:Bi']
        df['Na:Bi-F:Bi'] = df['Na:Bi'] - df['F:Bi']
        df['GeometricMean'] = np.sqrt(df['Na:Bi'] * df['F:Bi'])
        #df['sqrt_Na:Bi'] = np.sqrt(df['Na:Bi'])
        #df['sqrt_F:Bi'] = np.sqrt(df['F:Bi'])
        return df

    def perform_feature_engineering(self, poly_degree=3):
        """
        Perform feature engineering including polynomial feature expansion.

        Args:
            poly_degree (int): Degree for polynomial feature expansion
        """
        print("\nPerforming feature engineering...")
        print(f"Adding polynomial features (degree={poly_degree})")

        # Add basic engineered features
        self.X_train = self._add_features(self.X_train)
        self.X_test = self._add_features(self.X_test)

        # Create polynomial features (fit only on training data)
        self.poly = PolynomialFeatures(
            degree=poly_degree,
            include_bias=False,
            interaction_only=False
        )
        self.poly.fit(self.X_train[['Na:Bi', 'F:Bi']])

        # Transform both datasets
        poly_train = self.poly.transform(self.X_train[['Na:Bi', 'F:Bi']])
        poly_test = self.poly.transform(self.X_test[['Na:Bi', 'F:Bi']])
        self.poly_feature_names = self.poly.get_feature_names_out(['Na:Bi', 'F:Bi'])

        # Create DataFrames and drop base columns to avoid duplication
        poly_train_df = pd.DataFrame(poly_train, columns=self.poly_feature_names).drop(columns=['Na:Bi', 'F:Bi'])
        poly_test_df = pd.DataFrame(poly_test, columns=self.poly_feature_names).drop(columns=['Na:Bi', 'F:Bi'])

        # Combine features
        self.X_train = pd.concat([self.X_train, poly_train_df], axis=1)
        self.X_test = pd.concat([self.X_test, poly_test_df], axis=1)

        # Save engineered dataset
        engineered_data = pd.concat([self.X_train, self.X_test])
        engineered_data['Crystal_Type'] = pd.concat([self.y_train, self.y_test]).values
        engineered_data.to_csv(f'{self.output_dir}/feature_engineered_data.csv', index=False)
        print(f"Feature engineering complete. Original features: 2, New features: {self.X_train.shape[1]}")

    def normalize_data(self):
        """Normalize features using StandardScaler (fit only on training data)."""
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        # Save normalized data
        train_scaled = pd.DataFrame(self.X_train_scaled, columns=self.X_train.columns)
        train_scaled['Crystal_Type'] = self.y_train.values
        train_scaled.to_csv(f'{self.output_dir}/train_data_scaled.csv', index=False)

        test_scaled = pd.DataFrame(self.X_test_scaled, columns=self.X_test.columns)
        test_scaled['Crystal_Type'] = self.y_test.values
        test_scaled.to_csv(f'{self.output_dir}/test_data_scaled.csv', index=False)

    def train_model(self, param_grid=None, cv=5):
        """
        Train RandomForest classifier with hyperparameter optimization.

        Args:
            param_grid (dict): Hyperparameter grid for GridSearchCV
            cv (int): Number of cross-validation folds

        Returns:
            Optimized RandomForest classifier
        """
        print("\nStarting GridSearchCV hyperparameter optimization...")
        print(f"Using {cv}-fold cross-validation")

        # Default parameter grid if none provided
        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 150, 300, 450, 600, 650],
                'max_depth': [3, 5, 7, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 3, 5],
                'max_features': ['sqrt', 'log2']
            }

        # Create and run GridSearchCV
        grid_search = GridSearchCV(
            estimator=RandomForestClassifier(random_state=self.RANDOM_STATE),
            param_grid=param_grid,
            cv=cv,
            scoring='f1_weighted',
            n_jobs=8,  # Use all available cores
            verbose=1
        )

        start_time = time.time()
        grid_search.fit(self.X_train_scaled, self.y_train)
        optimization_time = time.time() - start_time

        print(f"GridSearchCV completed in {optimization_time:.2f} seconds!")
        print(f"Best F1 weighted score: {grid_search.best_score_:.4f}")

        # Save results
        cv_results = pd.DataFrame(grid_search.cv_results_)
        cv_results.to_csv(f'{self.output_dir}/gridsearch_results.csv', index=False)

        best_params = grid_search.best_params_
        pd.DataFrame([best_params]).to_csv(f'{self.output_dir}/best_parameters.csv', index=False)
        print(f"\nBest parameters: {best_params}")

        self.best_model = grid_search.best_estimator_

        return self.best_model

    def evaluate_model(self, set_name=""):
        """
        Evaluate model performance on training and test sets.

        Args:
            set_name (str): Optional prefix for evaluation results

        Returns:
            dict: Dictionary of evaluation metrics
        """
        results = {}

        for data_type, X, y in [("Train", self.X_train_scaled, self.y_train),
                                ("Test", self.X_test_scaled, self.y_test)]:
            y_pred = self.best_model.predict(X)
            y_proba = self.best_model.predict_proba(X)

            prefix = f"{set_name} " if set_name else ""
            results[f"{prefix}{data_type} Accuracy"] = accuracy_score(y, y_pred)
            results[f"{prefix}{data_type} Precision (Weighted)"] = precision_score(y, y_pred, average='weighted')
            results[f"{prefix}{data_type} Recall (Weighted)"] = recall_score(y, y_pred, average='weighted')
            results[f"{prefix}{data_type} F1 Macro"] = f1_score(y, y_pred, average='macro')
            results[f"{prefix}{data_type} F1 Weighted"] = f1_score(y, y_pred, average='weighted')

            # Class-specific metrics
            for class_id in [0, 1, 2]:
                class_name = ['Others', 'NaBiF4', 'BiF3'][class_id]
                results[f"{prefix}{data_type} Precision ({class_name})"] = precision_score(
                    y, y_pred, labels=[class_id], average=None)[0]
                results[f"{prefix}{data_type} Recall ({class_name})"] = recall_score(
                    y, y_pred, labels=[class_id], average=None)[0]
                results[f"{prefix}{data_type} F1 ({class_name})"] = f1_score(
                    y, y_pred, labels=[class_id], average=None)[0]

            if len(np.unique(y)) > 1:
                results[f"{prefix}{data_type} ROC AUC"] = roc_auc_score(
                    y, y_proba, multi_class='ovo', average='macro'
                )

        print("\nOptimized Model Performance:")
        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")

        return results

    def plot_feature_importance(self):
        """Visualize and save feature importance plot."""
        importance = self.best_model.feature_importances_
        sorted_idx = np.argsort(importance)[::-1]

        plt.figure(figsize=(12 / 2.54, 6 / 2.54))
        sns.barplot(x=importance[sorted_idx], y=self.X_train.columns[sorted_idx], palette='viridis')
        plt.title('')
        plt.xlabel('Importance Score')
        plt.ylabel('')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/feature_importance.png', dpi=600, bbox_inches='tight')

        # Save importance data
        imp_df = pd.DataFrame({
            'Feature': self.X_train.columns[sorted_idx],
            'Importance': importance[sorted_idx]
        })
        imp_df.to_csv(f'{self.output_dir}/feature_importance.csv', index=False)
        print("Feature importance saved")

    def plot_decision_boundary(self, filename="decision_boundary.png"):
        """
        Visualize the decision boundary of the classifier with smooth transitions.

        Args:
            filename (str): Output filename for the plot
        """
        # Create figure with proper size
        plt.figure(figsize=(3.15, 2.36), dpi=600)

        # Create symmetric grid
        na_min, na_max = self.data['Na:Bi'].min() * 0.1, self.data['Na:Bi'].max() * 1.1
        f_min, f_max = self.data['F:Bi'].min() * 0.1, self.data['F:Bi'].max() * 1.1

        # Create logarithmic grid
        na_range = np.logspace(np.log10(na_min), np.log10(na_max), 500)
        f_range = np.logspace(np.log10(f_min), np.log10(f_max), 500)

        # Create mesh grid
        na_grid, f_grid = np.meshgrid(na_range, f_range)
        grid_points = np.c_[na_grid.ravel(), f_grid.ravel()]
        grid_df = pd.DataFrame(grid_points, columns=['Na:Bi', 'F:Bi'])

        # Apply feature engineering
        grid_df = self._add_features(grid_df)
        poly_grid = self.poly.transform(grid_df[['Na:Bi', 'F:Bi']])
        poly_grid_df = pd.DataFrame(poly_grid, columns=self.poly_feature_names).drop(columns=['Na:Bi', 'F:Bi'])
        grid_full = pd.concat([grid_df, poly_grid_df], axis=1)

        # Normalize and predict
        grid_scaled = self.scaler.transform(grid_full)
        grid_pred = self.best_model.predict(grid_scaled)
        grid_proba = self.best_model.predict_proba(grid_scaled)

        # Reshape predictions
        Z = grid_pred.reshape(na_grid.shape)

        # Define color scheme
        colors = {
            0: '#FFA500',  # Orange (Type 0)
            1: '#FF4500',  # Red-orange (Type 1)
            2: '#87CEFA'  # Light blue (Type 2)
        }
        cmap = mpl.colors.ListedColormap([colors[0], colors[1], colors[2]])

        # Create smooth contour plot
        plt.contourf(
            na_grid,
            f_grid,
            Z,
            levels=[-0.5, 0.5, 1.5, 2.5],
            colors=list(colors.values()),
            alpha=0.9
        )

        # Plot training and test points
        plt.scatter(
            self.X_train['Na:Bi'],
            self.X_train['F:Bi'],
            c=self.y_train,
            edgecolor='black',
            cmap=cmap,
            s=12,
            linewidth=0.3,
            marker='o',
            alpha=0.9,
            label='Training Data',
            zorder=3
        )

        plt.scatter(
            self.X_test['Na:Bi'],
            self.X_test['F:Bi'],
            c=self.y_test,
            edgecolor='black',
            cmap=cmap,
            s=12,
            linewidth=0.3,
            marker='^',
            alpha=0.9,
            label='Test Data',
            zorder=3
        )

        # Create custom legend handles
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=colors[2], label=r'BiF$_3$'),
            Patch(facecolor=colors[1], label=r'NaBiF$_4$'),
            Patch(facecolor=colors[0], label='Others'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=5, label='Train'),
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=5, label='Test')
        ]

        # Add legend
        plt.legend(
            handles=legend_elements,
            loc='lower left',
            frameon=True,
            framealpha=0.9,
            fontsize=6,
            handlelength=1.5,
            borderpad=0.5,
            handletextpad=0.5
        )

        # Configure axes
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Na:Bi', fontsize=7)
        plt.ylabel('F:Bi', fontsize=7)

        # Adjust tick parameters
        plt.tick_params(axis='both', which='major', labelsize=6, pad=2)
        plt.tick_params(axis='both', which='minor', labelsize=5, pad=1)

        # Set proper margins
        plt.margins(0.02)

        # Finalize layout
        plt.tight_layout(pad=0.5)
        plt.savefig(f'{self.output_dir}/{filename}', dpi=1200, bbox_inches='tight')
        plt.close()
        print(f"Smooth decision boundary plot saved as '{filename}'")

        # Save grid data to CSV
        grid_df['Predicted_Class'] = grid_pred
        grid_df.to_csv(f'{self.output_dir}/decision_boundary_grid.csv', index=False)
        print("Decision boundary grid data saved to CSV")

    def full_evaluation(self, model_name="Optimized"):
        """
        Perform comprehensive model evaluation with detailed outputs.

        Args:
            model_name (str): Name for the model (used in output directories)
        """
        model_dir = os.path.join(self.output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        results = {}

        for data_type, X, y in [("Train", self.X_train_scaled, self.y_train),
                                ("Test", self.X_test_scaled, self.y_test)]:
            y_pred = self.best_model.predict(X)
            y_proba = self.best_model.predict_proba(X)

            # Classification report
            report = classification_report(y, y_pred, target_names=['Others', r'NaBiF$_4$', r'BiF$_3$'],
                                           output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            report_df.to_csv(f'{model_dir}/{data_type.lower()}_classification_report.csv')

            # Confusion matrix
            cm = confusion_matrix(y, y_pred)
            cm_df = pd.DataFrame(cm,
                                 index=['Others', r'NaBiF$_4$', r'BiF$_3$'],
                                 columns=['Others', r'NaBiF$_4$', r'BiF$_3$'])
            cm_df.to_csv(f'{model_dir}/{data_type.lower()}_confusion_matrix.csv')

            # Plot confusion matrix
            plt.figure(figsize=(8 / 2.54, 6 / 2.54))
            ax = plt.gca()
            disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                          display_labels=['Others', r'NaBiF$_4$', r'BiF$_3$'])
            disp.plot(ax=ax,
                      cmap='Blues',
                      values_format='d',
                      text_kw={'fontsize': 8}
                      )
            cbar = plt.gcf().axes[-1]
            cbar.shrink = 0.5
            cbar.aspect = 10
            ax.set_xlabel('Predicted Crystal Type', fontsize=7)
            ax.set_ylabel('True Crystal Type', fontsize=7)
            plt.title('')
            plt.tight_layout()
            plt.savefig(f'{model_dir}/{data_type.lower()}_confusion_matrix.png', dpi=600, bbox_inches='tight')
            plt.close()

            # Calculate metrics
            results[f"{data_type} Accuracy"] = accuracy_score(y, y_pred)
            results[f"{data_type} Precision (Weighted)"] = precision_score(y, y_pred, average='weighted')
            results[f"{data_type} Recall (Weighted)"] = recall_score(y, y_pred, average='weighted')
            results[f"{data_type} F1 Macro"] = f1_score(y, y_pred, average='macro')
            results[f"{data_type} F1 Weighted"] = f1_score(y, y_pred, average='weighted')
            results[f"{data_type} ROC AUC"] = roc_auc_score(y, y_proba, multi_class='ovo', average='macro')

            # ROC curves
            n_classes = 3
            y_bin = label_binarize(y, classes=[0, 1, 2])
            fpr, tpr, roc_auc = {}, {}, {}
            plt.figure(figsize=(7 / 2.54, 7 / 2.54))
            colors = {
                0: '#FFA500',  # Orange (Type 0)
                1: '#FF4500',  # Red-orange (Type 1)
                2: '#87CEFA'  # Light blue (Type 2)
            }
            crystals = ['Others', r'NaBiF$_4$', r'BiF$_3$']

            # Compute ROC for each class
            for i, color in zip(range(n_classes), colors):
                fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_proba[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
                plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
                         label=f'{crystals[i]} (AUC = {roc_auc[i]:.2f})')

            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('')
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(f'{model_dir}/{data_type.lower()}_roc_curves.png', dpi=600, bbox_inches='tight')
            plt.close()

            # Save ROC data
            roc_data = []
            for i in range(n_classes):
                for fp, tp in zip(fpr[i], tpr[i]):
                    roc_data.append({'Class': i, 'FPR': fp, 'TPR': tp, 'AUC': roc_auc[i]})
            pd.DataFrame(roc_data).to_csv(f'{model_dir}/{data_type.lower()}_roc_curve_data.csv', index=False)

        # Save all metrics
        pd.DataFrame.from_dict(results, orient='index', columns=['Value']).to_csv(
            f'{model_dir}/performance_metrics.csv'
        )

        return results

    def save_model(self):
        """Save trained model and preprocessing objects."""
        joblib.dump(self.best_model, f'{self.output_dir}/optimized_model.pkl')
        joblib.dump(self.scaler, f'{self.output_dir}/scaler.pkl')
        joblib.dump(self.poly, f'{self.output_dir}/poly_transformer.pkl')


# Main execution
if __name__ == "__main__":
    start_time = time.time()

    # Initialize classifier
    classifier = CrystalTypeClassifier(
        data_path='data_20.csv',
        output_dir="data_20",
        random_state=42
    )

    # Execute pipeline
    classifier.perform_feature_engineering(poly_degree=3)
    classifier.normalize_data()
    classifier.train_model()
    classifier.evaluate_model()
    classifier.plot_feature_importance()
    classifier.plot_decision_boundary("decision_boundary.png")
    classifier.full_evaluation()
    classifier.save_model()

    # Final report
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    print("Analysis completed! All results saved to:", classifier.output_dir)