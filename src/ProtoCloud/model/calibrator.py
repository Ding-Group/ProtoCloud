import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss



class simCalibration():
    def __init__(self):
        self.calibrators = {}
        self.global_calibrator = None
        self.cell_type_stats = {}
    
    def save(self, file_dir):
        """Save the trained calibrator to a file."""
        with open(file_dir + "calibrator_model.pkl", 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, file_dir):
        """Load a trained calibrator from a file."""
        with open(file_dir + "calibrator_model.pkl", 'rb') as f:
            return pickle.load(f)
        
    # def _create_ground_truth(self, df, pred_column='protoCloud'):
    #     """Create binary ground truth: 1 if prediction is correct, 0 otherwise"""
    #     if pred_column in df.columns:
    #         return (df[pred_column] == df['Actual']).astype(int)
    #     else:
    #         raise ValueError(f"Prediction column not found: {pred_column}")
    
    # def fit(self, df, similarity_col='sim_score', celltype_col='Actual', pred_column='protoCloud'):
    def fit(self, similarity_score, true_labels, pred_labels):
        # ground truth
        y_true = (pred_labels == true_labels).astype(int)
        X = similarity_score
        cell_types = true_labels
        min_samples_per_type = len(cell_types) / np.unique(cell_types).size // 2
        
        # global calibrator
        # self.global_calibrator = IsotonicRegression(out_of_bounds='clip')
        self.global_calibrator = IsotonicRegression()
        self.global_calibrator.fit(X, y_true)
        
        cell_type_counts = pd.Series(cell_types).value_counts()
        
        for cell_type in cell_type_counts.index:
            mask = cell_types == cell_type
            X_type = X[mask]
            y_type = y_true[mask]
            
            self.cell_type_stats[cell_type] = {
                'count': len(X_type),
                'accuracy': y_type.mean(),
                'mean_similarity': X_type.mean(),
                'std_similarity': X_type.std()
            }
            
            # cell type independent calibrator
            if len(X_type) >= min_samples_per_type:
                try:
                    calibrator = IsotonicRegression(out_of_bounds='clip')
                    calibrator.fit(X_type, y_type)
                    self.calibrators[cell_type] = calibrator
                except:
                    print(f"✗ {cell_type}: calibration failed, using global")
                    self.calibrators[cell_type] = 'global'
            else:
                print(f"○ {cell_type}: {len(X_type)} samples - using global calibrator")
                self.calibrators[cell_type] = 'global'
    

    def predict_proba(self, similarity_score, pred_labels):
        """
        Returns:
        --------
        calibrated_certainty : array
        """
        X = similarity_score
        cell_types = pred_labels
        
        calibrated_certainty = np.zeros(len(X))
        
        for i, (sim_score, cell_type) in enumerate(zip(X, cell_types)):
            if cell_type in self.calibrators:
                if self.calibrators[cell_type] == 'global':
                    calibrated_certainty[i] = self.global_calibrator.predict([sim_score])[0]
                else:
                    calibrated_certainty[i] = self.calibrators[cell_type].predict([sim_score])[0]
            else:
                # unseen cell type: global calibrator
                calibrated_certainty[i] = self.global_calibrator.predict([sim_score])[0]

        return calibrated_certainty
    

    def evaluate_calibration(self, similarity_score, true_labels, pred_labels):
        """
        Use Brier score to evaluate calibration performance.
        Returns:
        --------
        results : dict"""
        y_true = (pred_labels == true_labels).astype(int)
        y_prob_calibrated = self.predict_proba(similarity_score, pred_labels)
        
        try:
            brier_original = brier_score_loss(y_true, similarity_score)
            brier_calibrated = brier_score_loss(y_true, y_prob_calibrated)
            ece_org = self.expected_calibration_error(similarity_score, y_true, n_bins=20)
            ece_cal = self.expected_calibration_error(y_prob_calibrated, y_true, n_bins=20)
            
            print(f"Original Brier Score: {brier_original:.4f}, ECE: {ece_org:.4f}")
            print(f"Calibrated Brier Score: {brier_calibrated:.4f}, ECE: {ece_cal:.4f}")
            print(f"Brier Improvement: {brier_original - brier_calibrated}")
            print(f"ECE Improvement: {ece_org - ece_cal}")

            results = {
                'brier_score_original': brier_original,
                'brier_score_calibrated': brier_calibrated,
                'ece_original': ece_org,
                'ece_calibrated': ece_cal,
                'brier_improvement': brier_original - brier_calibrated,
                'ece_improvement': ece_org - ece_cal
            }
            return results
        
        except Exception as e:
            print(f"Evaluation failed: {e}")
            return None
    

    @staticmethod
    def expected_calibration_error(probs, labels, n_bins=20):
        bins  = np.linspace(0.0, 1.0, n_bins+1)
        idx   = np.digitize(probs, bins[1:-1], right=True)
        ece   = 0.0
        for k in range(n_bins):
            mask = idx == k
            if mask.any():
                acc  = labels[mask].mean()
                conf = probs[mask].mean()
                ece += np.abs(acc - conf) * mask.mean()
        return ece
    
    def get_cell_type_stats(self):
        return pd.DataFrame(self.cell_type_stats).T

    def extract_features(self, sim_matrix, top_k=2):
        """
        Returns:
        --------
        features : numpy.array
        """
        n_cells, n_prototypes = sim_matrix.shape

        if top_k < n_prototypes:
            # Top-k similarities
            features = np.sort(sim_matrix, axis=1)[:, -top_k:]
            self.feature_names = [f'top_{i+1}_sim' for i in range(top_k)]
        else:
            features = sim_matrix
            self.feature_names = [f'proto_{i}_sim' for i in range(n_prototypes)]
            
        return features


