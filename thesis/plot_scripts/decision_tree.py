import pandas as pd
from sklearn import tree
import graphviz


if __name__ == '__main__':
    df = pd.read_csv('tree_sample_separation.csv', index_col=0)
    features = ['width', 'length', 'intensity']
    X = df[features]
    y = df['particle_type']
    clf = tree.DecisionTreeClassifier(max_depth=3)
    clf.fit(X, y)
    print('Mean accuracy on training data: ', clf.score(X,y))
    dot_data = tree.export_graphviz(
        clf,
        out_file=None,
        feature_names=features,
        class_names=['proton', 'gamma'],
        rounded=True,
        rotate=True)

    graph = graphviz.Source(dot_data)
    graph.render("../Plots/decision_tree")
