#GBM provides a simple interface to best practices on using a GBM.
import random
import xgboost as xgb
from sklearn import linear_model
import category_encoders as ce

def buildGBM(X_train, X_test, y_train, y_test):
    best = float("inf")
    model = None
    for depth in range(3,6):#range(3,14):  #todo:make this a parameter
        print("Depth:" + str(depth))
        xg_reg = xgb.XGBRegressor(objective='reg:squarederror', 
                                colsample_bytree=0.8, 
                                learning_rate = 0.1, 
                                max_depth=depth, 
                                alpha = 1,
                                #gamma = 100,
                                #reg_lambda=10,
                                n_estimators = 500)
        xg_reg.fit(X_train, y_train)
        y_predT = xg_reg.predict(X_train)
        preds = xg_reg.predict(X_test)
        ll.accuracy(y_train, y_predT, "TRAIN")
        rmse, mse, r2 = ll.accuracy(y_test, preds)
        #rmse = np.sqrt(mean_squared_error(y_test, preds))
        #print("RMSE: " + str(rmse))
        #print(math.sqrt(rmse))
        if rmse < best:
            best = rmse
            model = xg_reg
    return model