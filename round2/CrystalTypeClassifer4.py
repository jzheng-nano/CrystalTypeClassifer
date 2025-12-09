import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, RocCurveDisplay, roc_auc_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier
import os
import warnings
import joblib
import time
import traceback


class ModelAnalysisSuite:
    def __init__(self, pipeline, output_dir, X_train, y_train, X_test, y_test):
        self.pipeline = pipeline
        self.output_dir = output_dir
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test
        if 'poly' in self.pipeline.named_steps:
            self.poly_transformer = self.pipeline.named_steps['poly']
            self.feature_names_after_poly = self.poly_transformer.get_feature_names_out(X_train.columns)
        else:
            self.feature_names_after_poly = X_train.columns

        self.X_full_orig = pd.concat([self.X_train, self.X_test])
        self._setup_plotting()

    def _setup_plotting(self):
        plt.rcParams['font.size'] = 7
        plt.rcParams['axes.labelsize'] = 7
        plt.rcParams['xtick.labelsize'] = 7
        plt.rcParams['ytick.labelsize'] = 7

    def run_all_analyses(self):
        print("\n--- Generating All Plots and Reports for the Best Model ---")
        self.plot_feature_importance()
        self.plot_decision_boundary()
        self.generate_evaluation_reports()
        print("--- All plots and reports for the best model have been generated. ---")

    def plot_feature_importance(self):
        print("  - Plotting feature importance...")
        model = self.pipeline.named_steps['model']
        if not (hasattr(model, 'feature_importances_') or hasattr(model, 'coef_')): return

        importances = model.feature_importances_ if hasattr(model, 'feature_importances_') else np.abs(
            self.pipeline.named_steps['model'].coef_[0])

        imp_df = pd.DataFrame({'feature': self.feature_names_after_poly, 'importance': importances}).sort_values(
            'importance', ascending=False)

        plt.figure(figsize=(12 / 2.54, max(6, len(self.feature_names_after_poly) * 0.6) / 2.54))
        sns.barplot(x='importance', y='feature', data=imp_df, palette='viridis')
        plt.xlabel('Importance Score')
        plt.ylabel('')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/best_model_feature_importance.png', dpi=600, bbox_inches='tight',
                    pad_inches=0.02)
        plt.close()

    def plot_decision_boundary(self):
        print("  - Plotting decision boundary...")
        x_feature, y_feature = 'Na:Bi', 'F:Bi'
        if x_feature not in self.X_train.columns or y_feature not in self.X_train.columns:
            print(f"    - SKIPPING: Decision boundary plot requires '{x_feature}' and '{y_feature}' columns.")
            return

        colors = {0: '#FFA500', 1: '#FF4500'}
        x_min, x_max = self.X_full_orig[x_feature].min() * 0.5, self.X_full_orig[x_feature].max() * 1.1
        y_min, y_max = self.X_full_orig[y_feature].min() * 0.5, self.X_full_orig[y_feature].max() * 1.1
        xx, yy = np.meshgrid(np.logspace(np.log10(x_min), np.log10(x_max), 200),
                             np.logspace(np.log10(y_min), np.log10(y_max), 200))

        if 'Temperature' in self.X_train.columns:
            temps_to_plot = sorted(self.X_full_orig['Temperature'].unique())
        else:
            temps_to_plot = [None]

        for temp in temps_to_plot:
            plt.figure(figsize=(3.15, 2.36), dpi=600)

            grid_data = {x_feature: xx.ravel(), y_feature: yy.ravel()}

            if temp is not None:
                grid_data['Temperature'] = temp
                grid_df_orig = pd.DataFrame(grid_data)[self.X_train.columns]
                train_mask = self.X_train.get('Temperature') == temp
                test_mask = self.X_test.get('Temperature') == temp
                filename = f'decision_boundary_at_{temp}C.png'
            else:
                grid_df_orig = pd.DataFrame(grid_data)[self.X_train.columns]
                train_mask = pd.Series([True] * len(self.X_train), index=self.X_train.index)
                test_mask = pd.Series([True] * len(self.X_test), index=self.X_test.index)
                filename = 'decision_boundary.png'

            Z = self.pipeline.predict(grid_df_orig).reshape(xx.shape)

            plt.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], colors=list(colors.values()), alpha=0.5)

            plt.scatter(self.X_train[train_mask][x_feature], self.X_train[train_mask][y_feature],
                        c=[colors[i] for i in self.y_train[train_mask]], edgecolor='black', s=12, lw=0.3,
                        marker='o', zorder=3)
            plt.scatter(self.X_test[test_mask][x_feature], self.X_test[test_mask][y_feature],
                        c=[colors[i] for i in self.y_test[test_mask]], edgecolor='black', s=12, lw=0.3, marker='^',
                        zorder=3)
            self._finalize_plot(filename)

    def _finalize_plot(self, filename):
        from matplotlib.patches import Patch
        colors = {0: '#FFA500', 1: '#FF4500'}
        legend_elements = [Patch(facecolor=colors[0], label='Others'), Patch(facecolor=colors[1], label=r'NaBiF$_4$'),
                           plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', ms=5,
                                      label='Train'),
                           plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', ms=5,
                                      label='Test')]
        plt.legend(handles=legend_elements, loc='lower left', frameon=True, fontsize=6)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Na:Bi')
        plt.ylabel('F:Bi')
        plt.tight_layout(pad=0.5)
        plt.savefig(os.path.join(self.output_dir, filename), dpi=600, bbox_inches='tight', pad_inches=0.02)
        plt.close()
        print(f"    - Saved plot: {filename}")

    def generate_evaluation_reports(self):
        print("  - Generating classification reports and confusion matrices...")
        model_dir = os.path.join(self.output_dir, "final_model_reports")
        os.makedirs(model_dir, exist_ok=True)
        for data_type, X, y in [("Full_Train_Set", self.X_train, self.y_train),
                                ("Final_HoldOut_Test_Set", self.X_test, self.y_test)]:
            y_pred = self.pipeline.predict(X)
            pd.DataFrame(classification_report(y, y_pred, target_names=['NaBiF4', 'Others'],
                                               output_dict=True)).transpose().to_csv(
                f'{model_dir}/{data_type}_report.csv')

            fig, ax = plt.subplots(figsize=(8 / 2.54, 6 / 2.54))
            ConfusionMatrixDisplay.from_predictions(y, y_pred, display_labels=['NaBiF4', 'Others'], cmap='Blues', ax=ax,
                                                    text_kw={'fontsize': 8})
            ax.set_xlabel('Predicted crystal type')
            ax.set_ylabel('True crystal type')
            plt.tight_layout()
            plt.savefig(f'{model_dir}/{data_type}_cm.png', dpi=600, bbox_inches='tight', pad_inches=0.02)
            plt.close()

            if hasattr(self.pipeline.named_steps['model'], "predict_proba"):
                y_proba = self.pipeline.predict_proba(X)[:, 1]
                roc_auc_score(y, y_proba)

                fig, ax = plt.subplots(figsize=(7 / 2.54, 7 / 2.54))
                RocCurveDisplay.from_estimator(self.pipeline, X, y, ax=ax, color='#FF4500', lw=1, name=f'ROC curve')

                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.legend(loc="lower right")

                plt.tight_layout()
                plt.savefig(f'{model_dir}/{data_type}_roc.png', dpi=600, bbox_inches='tight', pad_inches=0.02)
                plt.close()
        print(f"    - Reports and plots saved in '{model_dir}'.")


class MasterWorkflow:
    def __init__(self, data_path, output_dir, fe_degree, random_state=42):
        self.data_path = data_path
        self.output_dir = output_dir
        self.RANDOM_STATE = random_state
        self.FE_degree = fe_degree
        self.FE_interaction_only = False
        self.FE_include_bias = False
        os.makedirs(self.output_dir, exist_ok=True)
        warnings.filterwarnings("ignore")
        self.data = pd.read_csv(self.data_path)
        self._define_models_and_grids()

    def _define_models_and_grids(self):
        self.models_to_compare = {
             "Random Forest": RandomForestClassifier(random_state=self.RANDOM_STATE, class_weight='balanced'),
             "SVM": SVC(probability=True, random_state=self.RANDOM_STATE, class_weight='balanced'),
             "Logistic Regression": LogisticRegression(random_state=self.RANDOM_STATE, class_weight='balanced',
                                                       max_iter=5000),
            "Decision Tree": DecisionTreeClassifier(random_state=self.RANDOM_STATE, class_weight='balanced'),
            "Gradient Boosting": GradientBoostingClassifier(random_state=self.RANDOM_STATE),
             "AdaBoost": AdaBoostClassifier(random_state=self.RANDOM_STATE),
             "Gaussian NB": GaussianNB(),
             "QDA": QuadraticDiscriminantAnalysis(),
             "XGBoost": XGBClassifier(random_state=self.RANDOM_STATE, use_label_encoder=False, eval_metric='logloss')
        }

        self.param_grids = {
             "Random Forest": {
                 'model__n_estimators': [50, 100, 200, 300, 450, 650],
                 'model__max_depth': [2, 3, 5, 7, None],
                 'model__min_samples_split': [2, 3, 5, 10],
                 'model__min_samples_leaf': [1, 2, 3, 5, 8],
                 'model__max_features': ['sqrt', 'log2', None]
             },
             "SVM": [
                 {
                     'model__kernel': ['linear'],
                     'model__C': np.logspace(-3, 2, 10)
                 },
                 {
                     'model__kernel': ['rbf'],
                     'model__C': np.logspace(-3, 2, 10),
                     'model__gamma': ['scale', 'auto']
                 },
                 {
                     'model__kernel': ['poly'],
                     'model__C': np.logspace(-3, 2, 10),
                     'model__gamma': ['scale', 'auto'],
                     'model__degree': [2, 3, 4, 5],
                     'model__coef0': [-1, 0, 0.5, 1, 2]
                 },
                 {
                     'model__kernel': ['sigmoid'],
                     'model__C': np.logspace(-3, 2, 10),
                     'model__gamma': ['scale', 'auto'],
                     'model__coef0': [-1, 0, 0.5, 1, 2]
                 }
             ],
             "Logistic Regression": [
                 {
                     'model__penalty': ['l1'],
                     'model__C': np.logspace(-3, 2, 10),
                     'model__solver': ['liblinear', 'saga']
                 },
                 {
                     'model__penalty': ['l2'],
                     'model__C': np.logspace(-3, 2, 10),
                     'model__solver': ['liblinear', 'saga', 'lbfgs', 'newton-cg']
                 },
                 {
                     'model__penalty': ['elasticnet'],
                     'model__C': np.logspace(-3, 2, 10),
                     'model__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
                 },
                 {
                     'model__penalty': [None],
                     'model__solver': ['lbfgs', 'newton-cg', 'saga']
                 }
             ],
            "Decision Tree": {
                'model__criterion': ['gini', 'entropy','log_loss'],
                'model__max_depth': [2, 3, 5, 7, 10, None],
                'model__min_samples_split': [2, 3, 5, 10],
                'model__min_samples_leaf': [1, 2, 3, 4, 5],
                'model__max_features': ['sqrt', 'log2', None]
            },
            "Gradient Boosting": {
                'model__n_estimators': [100, 200, 300, 500],
                'model__learning_rate': [0.001, 0.01, 0.1],
                'model__max_depth': [2, 3, 4],
                'model__subsample': [0.8, 0.9],
                'model__min_samples_split': [4, 8, 10],
                'model__min_samples_leaf': [3, 4, 5],
                'model__max_features': ['sqrt', 'log2', None]
            },
             "AdaBoost": {
                 'model__n_estimators': [50, 100, 200, 300, 500],
                 'model__learning_rate': [0.001, 0.01, 0.1, 0.2, 1.0]
             },
             "Gaussian NB": {
                 'model__var_smoothing': np.logspace(-12, -2, 25)
             },
             "QDA": {
                 'model__reg_param': np.logspace(-6, 0, 15)
             },
             "XGBoost": {
                 'model__n_estimators': [100, 200, 300, 500],
                 'model__learning_rate': [0.01, 0.05, 0.1],
                 'model__max_depth': [2, 3, 7],
                 'model__subsample': [0.8, 0.9],
                 'model__colsample_bytree': [0.8, 0.9],
                 'model__reg_alpha': [0, 0.1, 1],
                 'model__reg_lambda': [0, 0.1, 1, 1],
                 'model__gamma': [0, 0.1, 0.5]
             }
        }

    def run_full_workflow(self):
        X = self.data.drop('Crystal_Type', axis=1)
        y = self.data['Crystal_Type']
        X_train_full, X_test_final, y_train_full, y_test_final = train_test_split(X, y, test_size=0.2, stratify=y,
                                                                                  random_state=self.RANDOM_STATE)

        print(
            f"\n--- Initial Hold-Out Split: {len(X_train_full)} training samples, {len(X_test_final)} final test samples ---")
        print(f"--- Using Polynomial Features with degree={self.FE_degree} ---")

        print(f"\n--- STEP 1: Robust Model Selection on {len(X_train_full)} Training Samples (Nested CV) ---")
        comparison_df = self._run_robust_model_selection(X_train_full, y_train_full)

        top_n_models = 3
        top_models_to_analyze = comparison_df.head(top_n_models)
        print(f"\n--- Identified Top {top_n_models} Models for Final Analysis ---")
        print(top_models_to_analyze.to_string(index=False))

        all_models_metrics = []

        for index, row in top_models_to_analyze.iterrows():
            model_name = row['Model']
            print("\n" + "=" * 80)
            print(f"######  ANALYZING MODEL: {model_name}  ######")
            print("=" * 80)

            print(
                f"\n--- STEP 2: Training Final '{model_name}' Pipeline on FULL {len(X_train_full)} Training Samples ---")
            
            model_analysis_dir = os.path.join(self.output_dir, f"{model_name.replace(' ', '_')}_analysis")
            os.makedirs(model_analysis_dir, exist_ok=True)
            
            final_pipeline = self._train_final_pipeline(X_train_full, y_train_full, model_name, model_analysis_dir)

            print(f"\n--- STEP 3: Generating All Analyses Using the Final '{model_name}' Model ---")
            analysis_suite = ModelAnalysisSuite(final_pipeline, model_analysis_dir, X_train_full, y_train_full, X_test_final,
                                                y_test_final)
            analysis_suite.run_all_analyses()

            def get_metrics_from_report(report_path, prefix):
                report_df = pd.read_csv(report_path, index_col=0)
                metrics = {
                    f'{prefix}_Accuracy': report_df.loc['accuracy', 'support'],
                    f'{prefix}_Macro_Avg_Precision': report_df.loc['macro avg', 'precision'],
                    f'{prefix}_Macro_Avg_Recall': report_df.loc['macro avg', 'recall'],
                    f'{prefix}_Macro_Avg_F1': report_df.loc['macro avg', 'f1-score'],
                    f'{prefix}_Weighted_Avg_Precision': report_df.loc['weighted avg', 'precision'],
                    f'{prefix}_Weighted_Avg_Recall': report_df.loc['weighted avg', 'recall'],
                    f'{prefix}_Weighted_Avg_F1': report_df.loc['weighted avg', 'f1-score']
                }
                return metrics

            train_report_path = f'{model_analysis_dir}/final_model_reports/Full_Train_Set_report.csv'
            test_report_path = f'{model_analysis_dir}/final_model_reports/Final_HoldOut_Test_Set_report.csv'

            train_metrics = get_metrics_from_report(train_report_path, 'Train')
            test_metrics = get_metrics_from_report(test_report_path, 'Test')
            
            model_summary_metrics = {
                "Model": model_name,
                "CV_Mean_F1": row['Mean CV F1'],
                "CV_Std_F1": row['Std CV F1']
            }
            model_summary_metrics.update(train_metrics)
            model_summary_metrics.update(test_metrics)
            all_models_metrics.append(model_summary_metrics)


            print(f"\n--- Final Performance of '{model_name}' on FULL Train Set ---")
            for key, value in train_metrics.items():
                print(f"  - {key}: {value:.4f}")

            print(f"\n--- Final Performance of '{model_name}' on UNSEEN Hold-Out Test Set ---")
            for key, value in test_metrics.items():
                print(f"  - {key}: {value:.4f}")
        
        return all_models_metrics

    def _run_robust_model_selection(self, X_train_full, y_train_full, n_splits=10):
        all_results = []
        for name, model in self.models_to_compare.items():
            start_time = time.time()

            pipeline = Pipeline([
                ('poly', PolynomialFeatures(degree=self.FE_degree, interaction_only=self.FE_interaction_only,
                                            include_bias=self.FE_include_bias)),
                ('scaler', StandardScaler()),
                ('model', model)
            ])

            inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.RANDOM_STATE)
            current_param_grid = self.param_grids.get(name, {})

            grid_search = GridSearchCV(pipeline, current_param_grid, cv=inner_cv, scoring='f1_weighted', n_jobs=-1)

            outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.RANDOM_STATE)

            try:
                scores = cross_val_score(grid_search, X_train_full, y_train_full, cv=outer_cv, scoring='f1_weighted',
                                         n_jobs=1)
                all_results.append({'Model': name, 'Mean CV F1': scores.mean(), 'Std CV F1': scores.std()})
                print(
                    f"  - {name}: Mean CV F1 = {scores.mean():.4f} (± {scores.std():.4f}) | Time: {time.time() - start_time:.2f}s")
            except Exception as e:
                print(f"  - {name}: FAILED - {str(e)}")
                all_results.append({'Model': name, 'Mean CV F1': 0, 'Std CV F1': 0})

        comparison_df = pd.DataFrame(all_results).sort_values(by='Mean CV F1', ascending=False)
        print("\n--- Nested CV Comparison Summary (on Training Set) ---")
        print(comparison_df.to_string(index=False))
        comparison_df.to_csv(f'{self.output_dir}/nested_cv_model_comparison.csv', index=False)
        
        if comparison_df.iloc[0]['Mean CV F1'] == 0:
            print("WARNING: The best model has a score of 0. Check for errors during cross-validation.")

        return comparison_df

    def _train_final_pipeline(self, X_train_full, y_train_full, best_model_name, output_dir):
        model_template = self.models_to_compare[best_model_name]

        pipeline = Pipeline([
            ('poly', PolynomialFeatures(degree=self.FE_degree, interaction_only=self.FE_interaction_only,
                                        include_bias=self.FE_include_bias)),
            ('scaler', StandardScaler()),
            ('model', model_template)
        ])
        current_param_grid = self.param_grids[best_model_name]

        cv_for_final_tuning = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.RANDOM_STATE)

        grid_search = GridSearchCV(pipeline, current_param_grid, cv=cv_for_final_tuning, scoring='f1_weighted', n_jobs=-1)
        grid_search.fit(X_train_full, y_train_full)

        final_pipeline = grid_search.best_estimator_
        joblib.dump(final_pipeline, f'{output_dir}/final_model_pipeline.pkl')

        best_params_df = pd.DataFrame([grid_search.best_params_])
        best_params_df.to_csv(f'{output_dir}/best_model_parameters.csv', index=False)

        print(f"Final pipeline saved to '{output_dir}/final_model_pipeline.pkl'")
        print(f"Best parameters: {grid_search.best_params_}")
        return final_pipeline


if __name__ == "__main__":
    scenarios = [
        {"name": "T=20_and_120", "data_path": "data_all.csv"}
        #{"name": "T=20", "data_path": "data_20.csv"},
        #{"name": "T=120", "data_path": "data_120.csv"}
    ]

    if not os.path.exists("data_all.csv"):
        print("Creating sample data file for testing...")
        sample_data = pd.DataFrame({
            'Na:Bi': np.random.uniform(0.1, 10, 100),
            'F:Bi': np.random.uniform(0.1, 10, 100),
            'Temperature': np.random.choice([20, 120], 100),
            'Crystal_Type': np.random.choice([0, 1], 100, p=[0.7, 0.3])
        })
        sample_data.to_csv("data_all.csv", index=False)
        print("Sample data file 'data_all.csv' created for testing.")

    final_summary = []

    fe_degrees_to_test = range(1, 3)

    for degree in fe_degrees_to_test:
        print("\n" + "#" * 80)
        print(f"######  STARTING EXPERIMENTS FOR FE_DEGREE = {degree}  ######")
        print("#" * 80)

        for scenario in scenarios:
            if not os.path.exists(scenario["data_path"]):
                print(f"\nSKIPPING: data file '{scenario['data_path']}' not found.")
                continue

            try:
                output_dir = f"results_{scenario['name']}_FE{degree}"

                workflow = MasterWorkflow(data_path=scenario["data_path"],
                                          output_dir=output_dir,
                                          fe_degree=degree)

                all_models_metrics = workflow.run_full_workflow()

                for model_metrics in all_models_metrics:
                    scenario_summary = {
                        "Scenario": scenario["name"],
                        "FE_Degree": degree,
                        "Best Model Type": model_metrics["Model"],
                        "Robust CV Mean F1": model_metrics['CV_Mean_F1'],
                        "Robust CV Std F1": model_metrics['CV_Std_F1']
                    }
                    scenario_summary.update({k: v for k, v in model_metrics.items() if k not in ["Model", "CV_Mean_F1", "CV_Std_F1"]})
                    final_summary.append(scenario_summary)

            except Exception as e:
                print(f"\nERROR running scenario '{scenario['name']}' for FE_Degree={degree}: {e}")
                traceback.print_exc()

    print("\n\n" + "=" * 80)
    print(" " * 28 + "FINAL SUMMARY OF ALL SCENARIOS")
    print("=" * 80)
    if final_summary:
        summary_df = pd.DataFrame(final_summary)
        summary_csv_path = "all_scenarios_summary.csv"
        summary_df.to_csv(summary_csv_path, index=False)
        print(summary_df.to_string(index=False))
        print(f"\nFinal summary saved to '{summary_csv_path}'")
    else:
        print("No experiments were successfully run.")
    print("=" * 80)
