import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    column_names = [
        "LOB_Peril_Region", "RecordID", "Loss", "Category", "Flag", "Ratio", "County_LOB"
    ]
    df = pd.read_csv(filepath, header=None, names=column_names)
    return df

def cap_outliers(series, cap_percentile=99):
    cap_value = np.percentile(series, cap_percentile)
    return np.where(series > cap_value, cap_value, series)

def drop_constant_columns(df):
    constant_cols = []
    for col in df.columns:
        if df[col].nunique() == 1:
            constant_cols.append(col)
    return df.drop(columns=constant_cols)

def handle_missing_values(df):
    df = df.dropna(axis=1, how='all')
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype in [np.float64, np.int64]:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
    return df

def encode_categoricals(df):
    for col in df.select_dtypes(include=['object']).columns:
        n_unique = df[col].nunique()
        if n_unique < 50:
            df[col + '_code'] = df[col].astype('category').cat.codes
        else:
            df[col + '_freq'] = df[col].map(df[col].value_counts())
    return df

def engineer_features(df):
    print("\nNew features from existing data created.")
    
    if 'County_LOB' in df.columns:
        df['County'] = df['County_LOB'].str.split('_').str[1]
        
        county_stats = df.groupby('County').agg({
            'Loss': ['mean', 'std', 'count', 'sum']
        }).reset_index()
        county_stats.columns = ['County', 'County_AvgLoss', 'County_StdLoss', 'County_EventCount', 'County_TotalLoss']
       

        county_stats['County_StdLoss'] = county_stats['County_StdLoss'].fillna(0)
        
        df = df.merge(county_stats, on='County', how='left')
        
        df['County_AvgLoss_log'] = np.log1p(df['County_AvgLoss'])
        df['County_TotalLoss_log'] = np.log1p(df['County_TotalLoss'])
        df['County_LossVolatility'] = df['County_StdLoss'] / (df['County_AvgLoss'] + 1)

    
    if 'LOB_Pure' in df.columns and 'Region' in df.columns:
        lob_region_stats = df.groupby(['LOB_Pure', 'Region']).agg({
            'Loss': ['mean', 'count']
        }).reset_index()
        lob_region_stats.columns = ['LOB_Pure', 'Region', 'LOB_Region_AvgLoss', 'LOB_Region_Count']
        
        df = df.merge(lob_region_stats, on=['LOB_Pure', 'Region'], how='left')
        df['LOB_Region_AvgLoss_log'] = np.log1p(df['LOB_Region_AvgLoss'])
        

    
    if 'Loss' in df.columns:
        df['Loss_Percentile_InRegion'] = df.groupby('Region')['Loss'].rank(pct=True)
        df['Loss_Percentile_InLOB'] = df.groupby('LOB_Pure')['Loss'].rank(pct=True)
        
        loss_95th = df['Loss'].quantile(0.95)
        loss_99th = df['Loss'].quantile(0.99)
        df['IsExtremeLoss_95'] = (df['Loss'] >= loss_95th).astype(int)
        df['IsExtremeLoss_99'] = (df['Loss'] >= loss_99th).astype(int)       

    
    if 'Ratio' in df.columns:
        df['Ratio_Category'] = pd.cut(df['Ratio'], bins=5, labels=['VeryLow', 'Low', 'Medium', 'High', 'VeryHigh'])
        df['Ratio_Category_code'] = df['Ratio_Category'].astype('category').cat.codes
        
        region_avg_ratio = df.groupby('Region')['Ratio'].transform('mean')
        df['Ratio_vs_RegionAvg'] = df['Ratio'] / (region_avg_ratio + 0.01)
        

    
    if 'County' in df.columns and 'Loss' in df.columns:
        region_total = df.groupby('Region')['Loss'].transform('sum')
        county_total = df.groupby(['Region', 'County'])['Loss'].transform('sum')
        df['County_LossShare_InRegion'] = county_total / (region_total + 1)
        
  

    
    return df

def clean_data(df):
    df = drop_constant_columns(df)
    df = handle_missing_values(df)
    
    if 'Loss' in df.columns:
        df['Loss_capped'] = cap_outliers(df['Loss'], 99)
        df['Loss_log'] = np.log1p(df['Loss_capped'])
    
    if 'LOB_Peril_Region' in df.columns:
        df['LOB_Pure'] = df['LOB_Peril_Region'].str.split('_').str[-1]
        df['Peril'] = df['LOB_Peril_Region'].str.split('_').str[1]
        df['Region'] = df['LOB_Peril_Region'].str.split('_').str[2]
    
    df = encode_categoricals(df)
    df = engineer_features(df)
    
    return df

def _select_clustering_features(df, features):
    features = [f for f in features if f in df.columns]
    
    to_remove = []
    for f in features:
        if df[f].nunique() <= 1:
            to_remove.append(f)
    
    if to_remove:
        print(f"Removing constant features: {', '.join(to_remove)}")
        features = [f for f in features if f not in to_remove]
    
    return df[features].copy()

def _scale_features(df_features, scaler=None):
    if scaler is None:
        scaler = StandardScaler()
    X = df_features.values.astype(float)
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def prepare_features_for_clustering(df, features, scale=True):
    features_df = _select_clustering_features(df, features)
    if features_df.empty:
        raise ValueError('No valid features found for clustering.')

    print(f"\nUsing features: {', '.join(features_df.columns)}")
    print(f"{len(features_df.columns)} features, {len(features_df)} observations")
    
    nan_counts = features_df.isnull().sum()
    if nan_counts.any():
        print("\n NaN values found:")
        for col, count in nan_counts[nan_counts > 0].items():
            print(f"  {col}: {count} NaNs ({count/len(features_df)*100:.1f}%)")
        print("Filling with median values...")
        features_df = features_df.fillna(features_df.median())
  
    corr_matrix = features_df.corr()
    
    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.7:
                high_corr.append({
                    'f1': corr_matrix.columns[i],
                    'f2': corr_matrix.columns[j],
                    'corr': corr_matrix.iloc[i, j]
                })
    
    if high_corr:
        print("\nWarning - high correlation:")
        for pair in high_corr:
            print(f"  {pair['f1']} <-> {pair['f2']}: {pair['corr']:.3f}")
    
    if scale:
        X_scaled, scaler = _scale_features(features_df)
        return X_scaled
    else:
        return features_df.values.astype(float)

def run_eda(df):
    print("DataFrame Info:")
    print(df.info())
    print(df.describe(include='all'))
    
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 1:
        correlation_matrix = numeric_df.corr()
        print("\nCorrelation Matrix:")
        print(correlation_matrix)   
    
        print("\nHighly Correlated Feature Pairs (>0.9):")
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                if abs(correlation_matrix.iloc[i, j]) > 0.9:
                    high_corr_pairs.append({
                        'Feature 1': correlation_matrix.columns[i],
                        'Feature 2': correlation_matrix.columns[j],
                        'Correlation': correlation_matrix.iloc[i, j]
                    })
        if high_corr_pairs:
            for pair in high_corr_pairs:
                print(f"  {pair['Feature 1']} <-> {pair['Feature 2']}: {pair['Correlation']:.3f}")
                plt.figure(figsize=(10, 8))
                plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
                plt.colorbar(label='Correlation')
                plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45, ha='right')
                plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
                plt.title('Correlation Heatmap')
                plt.tight_layout()
                plt.show()
        else:
            print("No highly correlated pairs found.")      

    if 'Loss' in df.columns:
        plt.hist(df['Loss'], bins=50, log=True)
        plt.xlabel('Event Loss')
        plt.ylabel('Frequency')
        plt.title('Distribution of Event Losses (Original)')
        plt.show()    

    if 'Loss_log' in df.columns:
        plt.hist(df['Loss_log'], bins=50)
        plt.xlabel('Log(Event Loss)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Log Transformed Event Losses')
        plt.show()
  
    categorical_cols = ['LOB_Peril_Region', 'Category', 'County_LOB']
    for col in categorical_cols:
        if col in df.columns:
            print(f"\nValue counts for {col}:")
            print(df[col].value_counts())