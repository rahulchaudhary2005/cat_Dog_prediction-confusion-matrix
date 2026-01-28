import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix,ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

# Load dataset
y_true = ['Cat'] * 10 + ['Dog'] * 12 + ['Horse'] * 10
y_pred = ['Cat'] * 8 + ['Dog'] + ['Horse'] + ['Cat'] * 2 + ['Dog'] * 10 + ['Horse'] * 8 + ['Dog'] * 2
classes = ['Cat', 'Dog', 'Horse']

# Encode labels
le = LabelEncoder()
y_true_encoded = le.fit_transform(y_true)
y_pred_encoded = le.transform(y_pred)
class_names = le.classes_

print("The data is ",y_true_encoded)
print("The predicted data is ",y_pred_encoded)
print("The class names are ",class_names)

# generate the confusion matrix
cm=confusion_matrix(y_true_encoded, y_pred_encoded)
print("Confusion Matrix:\n", cm)

# Plot confusion matrix


# plt.figure(figsize=(8, 6))
# plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
# plt.title('Confusion Matrix')
# plt.colorbar()
# tick_marks = np.arange(len(class_names))
# plt.xticks(tick_marks, class_names, rotation=45)
# plt.yticks(tick_marks, class_names)
# plt.ylabel('True label')
# plt.xlabel('Predicted label')
# thresh = cm.max() / 2.
# for i, j in np.ndindex(cm.shape):
#     plt.text(j, i, format(cm[i, j], 'd'),
#              horizontalalignment="center",
#              color="white" if cm[i, j] > thresh else "black")
# plt.tight_layout()
# plt.show()

cm = confusion_matrix(y_true, y_pred, labels=classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(cmap=plt.cm.Reds, values_format='d')
plt.title('Confusion Matrix', fontsize=30, pad=10)
plt.xlabel('Prediction', fontsize=20)
plt.ylabel('Actual', fontsize=20)
plt.gca().xaxis.set_label_position('top')
plt.gca().xaxis.tick_top()
plt.gca().figure.subplots_adjust(bottom=0.2)
plt.gca().figure.text(0.5, 0.05, 'Prediction', ha='center', fontsize=13)

plt.show()