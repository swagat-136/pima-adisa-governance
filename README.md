# Fast ADISA Demo & Machine Unlearning

This Streamlit application demonstrates a fast, lightweight Digit Classifier using an ensemble of specialized Neural Network models (FastCNN). It features instant predictions on handwritten images and an integrated lab to experiment with **Machine Unlearning**.

## How Machine Unlearning is demonstrated in this app (SISA)

Machine Unlearning is the process of completely removing the influence of a specific training sample from a deployed model without the massive computational expense of retraining the model from scratch.

This app implements a **SISA** (Sharded, Isolated, Sliced, and Aggregated) paradigm:
1. **Sharding**: The training data is split into 5 entirely separate, disjoint chunks.
2. **Isolation**: We train 5 distinct "Expert" models, one on each chunk.
3. **Tracking**: The app maintains a dictionary (`sample_to_expert`) mapping exactly which data point went to which expert.
4. **Targeted Unlearning**: When the user requests to forget a sample (e.g., ID `42`), the system queries the mapping to find the affected expert. It removes the sample from that expert's local array and retrains *only that specific expert*.
5. **Efficiency**: The other 4 experts are completely untouched. This visualizes a massive speedup comparing the full ensemble retrain time vs the single-expert isolated retrain time. The Streamlit UI measures this execution time in realtime and generates a graph of the performance delta.
