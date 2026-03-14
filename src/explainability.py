import shap
import matplotlib.pyplot as plt


def explain_model(model, X_sample):

    print("\nGenerating SHAP explanations...")

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_sample)

    # Summary plot
    shap.summary_plot(shap_values, X_sample)

    # Bar plot
    shap.summary_plot(shap_values, X_sample, plot_type="bar")