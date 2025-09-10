# stock_predictor.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

class AdvancedStockPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.results = {}
        self.feature_importance = {}
        
    def load_and_preprocess_data(self, file_path, symbol=None):
        """Load and preprocess the stock data"""
        print("Loading and preprocessing data...")
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Filter for specific symbol if provided
        if symbol:
            df = df[df['Symbol'] == symbol]
            print(f"Filtered data for symbol: {symbol}")
        
        # Sort by date
        df = df.sort_values(['Symbol', 'Date']).reset_index(drop=True)
        
        # Create additional features
        df = self.create_advanced_features(df)
        
        print(f"Dataset shape: {df.shape}")
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"Symbols: {df['Symbol'].nunique()}")
        
        return df
    
    def create_advanced_features(self, df):
        """Create advanced technical features"""
        grouped = df.groupby('Symbol')
        
        # Price momentum features
        df['Price_Change'] = grouped['Close'].pct_change()
        df['Price_Gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        
        # Volume features
        df['Volume_Change'] = grouped['Volume'].pct_change()
        df['Volume_Price_Trend'] = df['Volume'] * df['Price_Change']
        
        # Trend features
        df['Trend_5'] = grouped['Close'].transform(lambda x: x.rolling(5).mean() / x.rolling(20).mean())
        df['Trend_20'] = grouped['Close'].transform(lambda x: x.rolling(20).mean() / x.rolling(50).mean())
        
        # Volatility features
        df['Range'] = (df['High'] - df['Low']) / df['Close']
        df['Volatility_5'] = grouped['Close'].transform(lambda x: x.rolling(5).std())
        df['Volatility_20'] = grouped['Close'].transform(lambda x: x.rolling(20).std())
        
        # Momentum indicators
        df['Momentum_5'] = grouped['Close'].transform(lambda x: x / x.shift(5) - 1)
        df['Momentum_20'] = grouped['Close'].transform(lambda x: x / x.shift(20) - 1)
        
        # Drop NaN values
        df = df.dropna()
        
        return df
    
    def prepare_features_target(self, df, target_horizon=5):
        """Prepare features and target variable"""
        features = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26', 'RSI_14', 'MACD',
            'BB_Upper', 'BB_Lower', 'Stochastic_14', 'Volume_MA_20',
            'Volume_Ratio', 'Daily_Return', 'Volatility_20',
            'Price_Change', 'Price_Gap', 'Volume_Change',
            'Volume_Price_Trend', 'Trend_5', 'Trend_20',
            'Range', 'Volatility_5', 'Volatility_20',
            'Momentum_5', 'Momentum_20'
        ]
        
        # One-hot encode sectors
        sector_dummies = pd.get_dummies(df['Sector'], prefix='Sector')
        features += list(sector_dummies.columns)
        
        X = pd.concat([df[features], sector_dummies], axis=1)
        
        # Target: future price after target_horizon days
        y = df.groupby('Symbol')['Close'].shift(-target_horizon)
        
        # Remove rows with NaN target
        valid_indices = y.notna()
        X = X[valid_indices]
        y = y[valid_indices]
        
        return X, y
    
    def train_models(self, X_train, y_train):
        """Train multiple advanced ML models"""
        print("Training advanced models...")
        
        # Define models with hyperparameters
        models = {
            'XGBoost': xgb.XGBRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, random_state=42
            ),
            'LightGBM': lgb.LGBMRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, random_state=42
            ),
            'Random Forest': RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
            ),
            'Ridge Regression': Ridge(alpha=1.0, random_state=42),
            'Lasso Regression': Lasso(alpha=0.1, random_state=42)
        }
        
        # Create pipelines with preprocessing
        for name, model in models.items():
            pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', RobustScaler()),
                ('model', model)
            ])
            
            print(f"Training {name}...")
            pipeline.fit(X_train, y_train)
            self.models[name] = pipeline
        
        print("All models trained successfully!")
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models"""
        print("\nEvaluating models...")
        
        results = {}
        for name, pipeline in self.models.items():
            y_pred = pipeline.predict(X_test)
            
            metrics = {
                'MAE': mean_absolute_error(y_test, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'R2': r2_score(y_test, y_pred),
                'MAPE': np.mean(np.abs((y_test - y_pred) / y_test)) * 100,
                'Accuracy_5%': np.mean(np.abs((y_test - y_pred) / y_test) < 0.05) * 100
            }
            
            results[name] = metrics
            print(f"\n{name}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
        
        self.results = results
        
        # Store feature importance for tree-based models
        for name, pipeline in self.models.items():
            if hasattr(pipeline.named_steps['model'], 'feature_importances_'):
                self.feature_importance[name] = pipeline.named_steps['model'].feature_importances_
        
        return results
    
    def plot_results(self, y_test, predictions, model_name):
        """Plot evaluation results"""
        plt.figure(figsize=(20, 12))
        
        # Actual vs Predicted
        plt.subplot(2, 2, 1)
        plt.scatter(y_test, predictions, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Prices')
        plt.ylabel('Predicted Prices')
        plt.title(f'{model_name} - Predictions vs Actual')
        
        # Error distribution
        plt.subplot(2, 2, 2)
        errors = y_test - predictions
        plt.hist(errors, bins=50, alpha=0.7)
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title('Error Distribution')
        
        # Time series comparison
        plt.subplot(2, 2, 3)
        sample_size = min(100, len(y_test))
        plt.plot(y_test.values[:sample_size], label='Actual', marker='o')
        plt.plot(predictions[:sample_size], label='Predicted', marker='x')
        plt.xlabel('Time Index')
        plt.ylabel('Price')
        plt.title('Actual vs Predicted (Sample)')
        plt.legend()
        
        # Feature importance
        if model_name in self.feature_importance:
            plt.subplot(2, 2, 4)
            feature_names = X_test.columns
            importance = self.feature_importance[model_name]
            indices = np.argsort(importance)[-10:]  # Top 10 features
            
            plt.barh(range(len(indices)), importance[indices])
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            plt.title('Top 10 Feature Importance')
        
        plt.tight_layout()
        plt.show()
    
    def predict_future(self, df, symbol, days_to_predict=10):
        """Predict future prices for a specific symbol"""
        best_model_name = min(self.results, key=lambda x: self.results[x]['RMSE'])
        best_model = self.models[best_model_name]
        
        print(f"\nUsing best model: {best_model_name}")
        
        # Get latest data for the symbol
        symbol_data = df[df['Symbol'] == symbol].tail(100)
        
        if symbol_data.empty:
            print(f"No data found for symbol {symbol}")
            return None
        
        # Prepare features for prediction
        X_current, _ = self.prepare_features_target(symbol_data)
        
        # Use the most recent data point
        latest_features = X_current.iloc[-1:].copy()
        
        future_predictions = []
        current_features = latest_features.copy()
        
        for day in range(days_to_predict):
            # Predict next price
            next_price = best_model.predict(current_features)[0]
            future_predictions.append(next_price)
            
            # Update features for next prediction (simplified approach)
            # In practice, you'd want a more sophisticated method
            current_features['Close'] = next_price
            current_features['Open'] = next_price * (1 + np.random.normal(0, 0.01))
            current_features['High'] = next_price * (1 + np.random.uniform(0, 0.02))
            current_features['Low'] = next_price * (1 - np.random.uniform(0, 0.015))
        
        return future_predictions

def main():
    # Initialize predictor
    predictor = AdvancedStockPredictor()
    
    # Load and preprocess data
    df = predictor.load_and_preprocess_data('stock_data_large.csv', symbol='STK_001')
    
    # Prepare features and target
    X, y = predictor.prepare_features_target(df, target_horizon=5)
    
    # Split data (time-series aware split)
    tscv = TimeSeriesSplit(n_splits=5)
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    print(f"Training set size: {len(X_train):,}")
    print(f"Test set size: {len(X_test):,}")
    
    # Train models
    predictor.train_models(X_train, y_train)
    
    # Evaluate models
    results = predictor.evaluate_models(X_test, y_test)
    
    # Make predictions with best model
    best_model_name = min(results, key=lambda x: results[x]['RMSE'])
    best_model = predictor.models[best_model_name]
    predictions = best_model.predict(X_test)
    
    # Plot results
    predictor.plot_results(y_test, predictions, best_model_name)
    
    # Predict future prices
    future_prices = predictor.predict_future(df, 'STK_001', days_to_predict=10)
    
    if future_predictions:
        current_price = df[df['Symbol'] == 'STK_001']['Close'].iloc[-1]
        print(f"\nCurrent price: ${current_price:.2f}")
        print(f"Predicted prices for next 10 days:")
        for i, price in enumerate(future_predictions, 1):
            change = ((price - current_price) / current_price) * 100
            print(f"Day {i}: ${price:.2f} ({change:+.2f}%)")

if __name__ == "__main__":
    main()