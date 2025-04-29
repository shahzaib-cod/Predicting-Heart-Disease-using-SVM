from flask import Flask, request, render_template
from model import load_and_explore_data, preprocess_data, perform_eda, feature_engineering_selection, model_selection_optimization, evaluate_model, create_app, plot_decision_boundary

if __name__ == "__main__":
   
    df = load_and_explore_data()
    
   
    df_scaled, scaler = preprocess_data(df)
    
   
    perform_eda(df_scaled)
    
   
    X_new, y, selected_features = feature_engineering_selection(df_scaled)
    
   
    best_model, X_train, X_test, y_train, y_test = model_selection_optimization(X_new, y)
    
   
    accuracy = evaluate_model(best_model, X_test, y_test, X_new, y)  
    
    plot_decision_boundary(best_model, X_new, y)

    
    app = create_app(best_model, scaler, accuracy) 
    app.run(debug=True, use_reloader=False)

