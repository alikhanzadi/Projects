import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split


def load_data_from_database(url, port, database, table, user, password):
    
    engine = create_engine("postgresql://{}:{}@{}:{}/{}".format(user, password, url, port, database))
    df = pd.read_sql("SELECT * FROM {}".format(table), con=engine)

    return df

def make_data_dict(X, y, test_size=.3, random_state=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, test_size=test_size)
    
    return {
        'X_train' : X_train,
        'X_test' : X_test,
        'y_train' : y_train,
        'y_test' : y_test,
        'random_state' : random_state,
        'test_size' : test_size
    }

def general_transformer(transformer, data_dict, random_state=None):
    
    if transformer == 'StandardScaler()':
        transformer.fit(data_dict['X_train'])
    else:
        transformer.fit(data_dict['X_train'], data_dict['y_train'])
        
    X_train = transformer.transform(data_dict['X_train'])
    X_test = transformer.transform(data_dict['X_test'])        
    y_train = data_dict['y_train']
    y_test = data_dict['y_test']
    return {
        'transformer' : transformer,
        'X_train' : X_train,
        'X_test' : X_test,
        'y_train' : y_train,
        'y_test' : y_test,
    }

def general_model(model, data_dict, random_state=None):
    model.fit(data_dict['X_train'], data_dict['y_train'])
    
    train_score = model.score(data_dict['X_train'], data_dict['y_train'])
    test_score = model.score(data_dict['X_test'], data_dict['y_test'])

    return {'model' : model,
            'train_score' : train_score,
            'test_score' : test_score
    }
