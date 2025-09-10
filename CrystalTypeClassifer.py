import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (confusion_matrix, classification_report,
                             f1_score, precision_score, recall_score, accuracy_score,
                             roc_auc_score, ConfusionMatrixDisplay, roc_curve, auc,
                             precision_recall_curve, average_precision_score)
from sklearn.preprocessing import StandardScaler
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
    A machine learning pipeline for binary classification of crystal types based on chemical ratios.

    Attributes:
        output_dir (str): Directory to save all outputs
        RANDOM_STATE (int): Random seed for reproducibility
        data (pd.DataFrame): Raw input data
        X_train, X_test, y_train, y_test: Split datasets
        best_model: Optimized RandomForest classifier
        scaler: Fitted StandardScaler
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
        print("Predicting target: 'Crystal_Type' (0=NaBiF4, 1=Others)")

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

    def perform_feature_engineering(self):
        """
        Placeholder method for feature engineering - no feature engineering is performed.
        This method is kept for compatibility with the original code structure.
        """
        print("\nSkipping feature engineering - using original features only")
        print("Using only 'Na:Bi' and 'F:Bi' features")

        # Save dataset without feature engineering
        original_data = pd.concat([self.X_train, self.X_test])
        original_data['Crystal_Type'] = pd.concat([self.y_train, self.y_test]).values
        original_data.to_csv(f'{self.output_dir}/original_data.csv', index=False)
        print("Using original 2 features without any engineering")

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
            scoring='f1',  # Changed to binary F1 score
            n_jobs=8,  # Use all available cores
            verbose=1
        )

        start_time = time.time()
        grid_search.fit(self.X_train_scaled, self.y_train)
        optimization_time = time.time() - start_time

        print(f"GridSearchCV completed in {optimization_time:.2f} seconds!")
        print(f"Best F1 score: {grid_search.best_score_:.4f}")

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

            # Binary classification metrics for each class
            results[f"{prefix}{data_type} Accuracy"] = accuracy_score(y, y_pred)
            results[f"{prefix}{data_type} Precision (NaBiF4)"] = precision_score(y, y_pred, pos_label=0)
            results[f"{prefix}{data_type} Recall (NaBiF4)"] = recall_score(y, y_pred, pos_label=0)
            results[f"{prefix}{data_type} F1 (NaBiF4)"] = f1_score(y, y_pred, pos_label=0)
            results[f"{prefix}{data_type} Precision (Others)"] = precision_score(y, y_pred, pos_label=1)
            results[f"{prefix}{data_type} Recall (Others)"] = recall_score(y, y_pred, pos_label=1)
            results[f"{prefix}{data_type} F1 (Others)"] = f1_score(y, y_pred, pos_label=1)
            results[f"{prefix}{data_type} Precision (Macro)"] = precision_score(y, y_pred, average='macro')
            results[f"{prefix}{data_type} Recall (Macro)"] = recall_score(y, y_pred, average='macro')
            results[f"{prefix}{data_type} F1 (Macro)"] = f1_score(y, y_pred, average='macro')
            results[f"{prefix}{data_type} Precision (Weighted)"] = precision_score(y, y_pred, average='weighted')
            results[f"{prefix}{data_type} Recall (Weighted)"] = recall_score(y, y_pred, average='weighted')
            results[f"{prefix}{data_type} F1 (Weighted)"] = f1_score(y, y_pred, average='weighted')

            # Binary ROC AUC
            results[f"{prefix}{data_type} ROC AUC"] = roc_auc_score(y, y_proba[:, 1])

            # Average precision score (better for imbalanced datasets)
            results[f"{prefix}{data_type} Average Precision"] = average_precision_score(y, y_proba[:, 1])

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

        # Normalize and predict
        grid_scaled = self.scaler.transform(grid_df)
        grid_pred = self.best_model.predict(grid_scaled)
        grid_proba = self.best_model.predict_proba(grid_scaled)

        # Reshape predictions
        Z = grid_pred.reshape(na_grid.shape)

        # Define color scheme for binary classification
        colors = {
            0: '#FF4500',  # Red-orange (NaBiF4)
            1: '#87CEFA'  # Light blue (Others)
        }
        cmap = mpl.colors.ListedColormap([colors[0], colors[1]])

        # Create smooth contour plot
        plt.contourf(
            na_grid,
            f_grid,
            Z,
            levels=[-0.5, 0.5, 1.5],
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
            Patch(facecolor=colors[0], label=r'NaBiF$_4$'),
            Patch(facecolor=colors[1], label='Others'),
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
        grid_df['Probability_NaBiF4'] = grid_proba[:, 0]
        grid_df['Probability_Others'] = grid_proba[:, 1]
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
            report = classification_report(y, y_pred, target_names=['NaBiF4', 'Others'],
                                           output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            report_df.to_csv(f'{model_dir}/{data_type.lower()}_classification_report.csv')

            # 提取macro和weighted平均指标
            macro_precision = report['macro avg']['precision']
            macro_recall = report['macro avg']['recall']
            macro_f1 = report['macro avg']['f1-score']

            weighted_precision = report['weighted avg']['precision']
            weighted_recall = report['weighted avg']['recall']
            weighted_f1 = report['weighted avg']['f1-score']

            # Confusion matrix
            cm = confusion_matrix(y, y_pred)
            cm_df = pd.DataFrame(cm,
                                 index=['NaBiF4', 'Others'],
                                 columns=['NaBiF4', 'Others'])
            cm_df.to_csv(f'{model_dir}/{data_type.lower()}_confusion_matrix.csv')

            # Plot confusion matrix
            plt.figure(figsize=(8 / 2.54, 6 / 2.54))
            ax = plt.gca()
            disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                          display_labels=['NaBiF4', 'Others'])
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
            results[f"{data_type} Precision (NaBiF4)"] = precision_score(y, y_pred, pos_label=0)
            results[f"{data_type} Recall (NaBiF4)"] = recall_score(y, y_pred, pos_label=0)
            results[f"{data_type} F1 (NaBiF4)"] = f1_score(y, y_pred, pos_label=0)
            results[f"{data_type} Precision (Others)"] = precision_score(y, y_pred, pos_label=1)
            results[f"{data_type} Recall (Others)"] = recall_score(y, y_pred, pos_label=1)
            results[f"{data_type} F1 (Others)"] = f1_score(y, y_pred, pos_label=1)
            results[f"{data_type} Precision (Macro)"] = macro_precision
            results[f"{data_type} Recall (Macro)"] = macro_recall
            results[f"{data_type} F1 (Macro)"] = macro_f1
            results[f"{data_type} Precision (Weighted)"] = weighted_precision
            results[f"{data_type} Recall (Weighted)"] = weighted_recall
            results[f"{data_type} F1 (Weighted)"] = weighted_f1

            results[f"{data_type} ROC AUC"] = roc_auc_score(y, y_proba[:, 1])
            results[f"{data_type} Average Precision"] = average_precision_score(y, y_proba[:, 1])

            # ROC curve for binary classification
            plt.figure(figsize=(7 / 2.54, 7 / 2.54))

            fpr, tpr, _ = roc_curve(y, y_proba[:, 1])
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, color='#FF4500', lw=2,
                     label=f'ROC curve (AUC = {roc_auc:.2f})')

            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('')
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(f'{model_dir}/{data_type.lower()}_roc_curve.png', dpi=600, bbox_inches='tight')
            plt.close()

            # Precision-Recall curve
            plt.figure(figsize=(7 / 2.54, 7 / 2.54))
            precision, recall, _ = precision_recall_curve(y, y_proba[:, 1])
            avg_precision = average_precision_score(y, y_proba[:, 1])

            plt.plot(recall, precision, color='#FF4500', lw=2,
                     label=f'Precision-Recall curve (AP = {avg_precision:.2f})')

            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.title('')
            plt.legend(loc="lower left")
            plt.tight_layout()
            plt.savefig(f'{model_dir}/{data_type.lower()}_precision_recall_curve.png', dpi=600, bbox_inches='tight')
            plt.close()

            # Save ROC and PR curve data
            roc_data = pd.DataFrame({'FPR': fpr, 'TPR': tpr, 'AUC': roc_auc})
            roc_data.to_csv(f'{model_dir}/{data_type.lower()}_roc_curve_data.csv', index=False)

            pr_data = pd.DataFrame({'Recall': recall, 'Precision': precision, 'AP': avg_precision})
            pr_data.to_csv(f'{model_dir}/{data_type.lower()}_precision_recall_data.csv', index=False)

        # Save all metrics
        pd.DataFrame.from_dict(results, orient='index', columns=['Value']).to_csv(
            f'{model_dir}/performance_metrics.csv'
        )

        return results

    def save_model(self):
        """Save trained model and preprocessing objects."""
        joblib.dump(self.best_model, f'{self.output_dir}/optimized_model.pkl')
        joblib.dump(self.scaler, f'{self.output_dir}/scaler.pkl')


# Main execution
if __name__ == "__main__":
    start_time = time.time()

    # Initialize classifier
    classifier = CrystalTypeClassifier(
        data_path='data_20.csv',
        output_dir="data_20",
        random_state=42
    )

    # Execute pipeline (without feature engineering)
    classifier.perform_feature_engineering()  # This now does nothing
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