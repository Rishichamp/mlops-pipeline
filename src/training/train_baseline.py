from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib, os, logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    logger.info(f'Test accuracy: {acc:.4f}')
    os.makedirs('artifacts', exist_ok=True)
    joblib.dump(model, 'artifacts/model.pkl')
    logger.info('Saved to artifacts/model.pkl')

if __name__ == '__main__':
    train()
