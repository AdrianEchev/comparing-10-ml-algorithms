"""
ANALYSIS OF 10 BINARY CLASSIFICATIO ALGORITHMS
@author: Adrián Echeverría P.
"""
def WriteFile(name, accuracy, precision, cm, f1, rmse, cv):
    file = "../output/results.txt"

    with open(file, "a") as output_file:
        output_file.write(f"Model: {name}\n")
        output_file.write(f"Accuracy: {accuracy}\n")
        output_file.write(f"Precision: {precision}\n")
        output_file.write(f"F1 Score: {f1}\n")
        output_file.write(f"RMSE: {rmse}\n")
        output_file.write(f"Cross-Validation: {cv}\n")
        output_file.write(f"Confusion Matrix: {cm}\n")
        output_file.write("\n")