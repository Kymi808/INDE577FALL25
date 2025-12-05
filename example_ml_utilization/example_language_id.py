"""
Example 4: Language Identification with RNN
============================================

This example demonstrates sequence classification using RNN from rice_ml.

The language ID dataset contains words from 5 European languages.
We use both bag-of-characters and RNN approaches.

Algorithms demonstrated:
- RNNClassifier (Recurrent Neural Network)
- GRUClassifier (Gated Recurrent Unit)
- Logistic Regression (baseline)
- Random Forest (baseline)

Prerequisites:
- lang_id.npz file in the data/ directory
"""

import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rice_ml import (
    # Preprocessing
    standardize,
    # Classifiers
    LogisticRegression,
    RandomForestClassifier,
    RNNClassifier,
    GRUClassifier,
    # Metrics
    accuracy_score,
    confusion_matrix,
)


def load_language_data(data_path="data/lang_id.npz"):
    """Load language identification dataset."""
    paths_to_try = [
        data_path,
        os.path.join(os.path.dirname(__file__), data_path),
        os.path.join(os.path.dirname(__file__), '..', 'data', 'lang_id.npz'),
        os.path.join(os.path.dirname(__file__), 'data', 'lang_id.npz'),
    ]
    
    for path in paths_to_try:
        if os.path.exists(path):
            data_path = path
            break
    else:
        raise FileNotFoundError(f"Could not find lang_id.npz. Tried: {paths_to_try}")
    
    with np.load(data_path) as data:
        return {
            'chars': data['chars'],
            'language_codes': data['language_codes'],
            'language_names': data['language_names'],
            'train_x': data['train_x'],
            'train_y': data['train_y'],
            'train_buckets': data['train_buckets'],
            'dev_x': data['dev_x'],
            'dev_y': data['dev_y'],
            'dev_buckets': data['dev_buckets'],
        }


def extract_char_histogram(X, n_chars):
    """Convert to bag-of-characters histogram."""
    n_samples = X.shape[0]
    histograms = np.zeros((n_samples, n_chars), dtype=float)
    
    for i in range(n_samples):
        chars = X[i][X[i] >= 0]
        if len(chars) > 0:
            for c in chars:
                histograms[i, c] += 1
            histograms[i] /= len(chars)
    
    return histograms


def train_rnn_batched(rnn, data, n_chars, epochs=25, batch_size=64):
    """
    Train RNN using the bucketed data structure.
    
    The lang_id dataset groups sequences by length in buckets for efficient batching.
    """
    train_x = data['train_x']
    train_y = data['train_y']
    train_buckets = data['train_buckets']
    n_languages = len(data['language_names'])
    
    # Compute bucket weights for sampling
    bucket_weights = train_buckets[:, 1] - train_buckets[:, 0]
    bucket_weights = bucket_weights / bucket_weights.sum()
    
    n_samples = train_x.shape[0]
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        
        for _ in range(n_samples // batch_size):
            # Sample a bucket weighted by size
            bucket_id = np.random.choice(len(bucket_weights), p=bucket_weights)
            start, end = train_buckets[bucket_id]
            
            # Sample batch from bucket
            indices = start + np.random.choice(end - start, size=min(batch_size, end - start), replace=False)
            
            batch_x = train_x[indices]
            batch_y = train_y[indices]
            
            # Convert to sequence format: list of (batch, n_chars) arrays
            # Find actual sequence length for this bucket
            seq_len = 0
            for j in range(batch_x.shape[1]):
                if np.all(batch_x[:, j] == -1):
                    break
                seq_len = j + 1
            
            xs = []
            for t in range(seq_len):
                x_t = np.eye(n_chars)[batch_x[:, t]]
                xs.append(x_t)
            
            # One-hot encode labels
            y_onehot = np.zeros((len(batch_y), n_languages))
            y_onehot[np.arange(len(batch_y)), batch_y] = 1
            
            loss = rnn.train_step(xs, y_onehot)
            epoch_loss += loss
            n_batches += 1
        
        avg_loss = epoch_loss / max(n_batches, 1)
        
        # Compute dev accuracy
        dev_acc = evaluate_rnn(rnn, data, n_chars, split='dev')
        print(f"  Epoch {epoch+1:2d}: loss={avg_loss:.4f}, dev_acc={dev_acc:.2%}")


def evaluate_rnn(rnn, data, n_chars, split='dev'):
    """Evaluate RNN on dev or test set."""
    if split == 'dev':
        X = data['dev_x']
        y = data['dev_y']
        buckets = data['dev_buckets']
    else:
        X = data['test_x']
        y = data['test_y']
        buckets = data['test_buckets']
    
    all_preds = []
    all_true = []
    
    for bucket_id in range(buckets.shape[0]):
        start, end = buckets[bucket_id]
        batch_x = X[start:end]
        batch_y = y[start:end]
        
        # Find sequence length
        seq_len = 0
        for j in range(batch_x.shape[1]):
            if np.all(batch_x[:, j] == -1):
                break
            seq_len = j + 1
        
        if seq_len == 0:
            continue
            
        xs = []
        for t in range(seq_len):
            x_t = np.eye(n_chars)[batch_x[:, t]]
            xs.append(x_t)
        
        preds = rnn.predict(xs)
        all_preds.extend(preds)
        all_true.extend(batch_y)
    
    return accuracy_score(np.array(all_true), np.array(all_preds))


def main():
    print("=" * 60)
    print("Language Identification with RNN")
    print("=" * 60)
    
    # Load data
    data = load_language_data()
    
    chars = data['chars']
    language_names = data['language_names']
    n_chars = len(chars)
    n_languages = len(language_names)
    
    print(f"\nDataset loaded:")
    print(f"  Languages: {list(language_names)}")
    print(f"  Alphabet size: {n_chars}")
    print(f"  Training samples: {len(data['train_x'])}")
    print(f"  Dev samples: {len(data['dev_x'])}")
    
    results = {}
    
    # =========================================================================
    # Baseline: Bag-of-Characters + Logistic Regression
    # =========================================================================
    print("\n" + "=" * 60)
    print("Baseline: Bag-of-Characters Features")
    print("=" * 60)
    
    X_train = extract_char_histogram(data['train_x'], n_chars)
    X_dev = extract_char_histogram(data['dev_x'], n_chars)
    y_train = data['train_y']
    y_dev = data['dev_y']
    
    X_train_std, params = standardize(X_train, return_params=True)
    X_dev_std = (X_dev - params["mean"]) / params["scale"]
    
    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(max_iter=500, learning_rate=0.5)
    lr.fit(X_train_std, y_train)
    y_pred = lr.predict(X_dev_std)
    results["Logistic Regression"] = accuracy_score(y_dev, y_pred)
    print(f"  Dev Accuracy: {results['Logistic Regression']:.2%}")
    
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(n_estimators=50, max_depth=20, random_state=42)
    rf.fit(X_train_std, y_train)
    y_pred = rf.predict(X_dev_std)
    results["Random Forest"] = accuracy_score(y_dev, y_pred)
    print(f"  Dev Accuracy: {results['Random Forest']:.2%}")
    
    # =========================================================================
    # RNN Classifier
    # =========================================================================
    print("\n" + "=" * 60)
    print("RNN Classifier (Sequence Model)")
    print("=" * 60)
    
    print(f"\nTraining RNN (hidden_size=128)...")
    np.random.seed(42)
    rnn = RNNClassifier(
        input_size=n_chars,
        hidden_size=128,
        output_size=n_languages,
        learning_rate=0.01
    )
    train_rnn_batched(rnn, data, n_chars, epochs=25, batch_size=32)
    results["RNN"] = evaluate_rnn(rnn, data, n_chars, split='dev')
    
    # =========================================================================
    # GRU Classifier
    # =========================================================================
    print("\n" + "=" * 60)
    print("GRU Classifier (Gated Recurrent Unit)")
    print("=" * 60)
    
    print(f"\nTraining GRU (hidden_size=128)...")
    np.random.seed(42)
    gru = GRUClassifier(
        input_size=n_chars,
        hidden_size=128,
        output_size=n_languages,
        learning_rate=0.01
    )
    train_rnn_batched(gru, data, n_chars, epochs=25, batch_size=32)
    results["GRU"] = evaluate_rnn(gru, data, n_chars, split='dev')
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("SUMMARY - Model Comparison")
    print("=" * 60)
    
    print(f"\n{'Model':<25} {'Dev Accuracy':>15}")
    print("-" * 42)
    for model, acc in sorted(results.items(), key=lambda x: -x[1]):
        print(f"{model:<25} {acc:>15.2%}")
    
    # Show confusion matrix for best RNN model
    best_rnn = gru if results.get("GRU", 0) >= results.get("RNN", 0) else rnn
    best_name = "GRU" if results.get("GRU", 0) >= results.get("RNN", 0) else "RNN"
    
    print(f"\nConfusion Matrix ({best_name}):")
    
    # Get predictions
    all_preds = []
    all_true = []
    buckets = data['dev_buckets']
    for bucket_id in range(buckets.shape[0]):
        start, end = buckets[bucket_id]
        batch_x = data['dev_x'][start:end]
        batch_y = data['dev_y'][start:end]
        
        seq_len = 0
        for j in range(batch_x.shape[1]):
            if np.all(batch_x[:, j] == -1):
                break
            seq_len = j + 1
        
        if seq_len == 0:
            continue
            
        xs = [np.eye(n_chars)[batch_x[:, t]] for t in range(seq_len)]
        preds = best_rnn.predict(xs)
        all_preds.extend(preds)
        all_true.extend(batch_y)
    
    cm = confusion_matrix(np.array(all_true), np.array(all_preds))
    
    print("     " + " ".join(f"{code:>4}" for code in data['language_codes']))
    for i, row in enumerate(cm):
        print(f"  {data['language_codes'][i]}: " + " ".join(f"{v:>4}" for v in row))
    


if __name__ == "__main__":
    main()