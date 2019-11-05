package ml.titanic;

import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

/**
 * Choose strategy to work with null data. Accuracy =  0.288
 */
public class Ex_2_Titanic_without_nulls {
    public static void main(String[] args) {
        SparkSession spark = TitanicUtils.getSparkSession();

        Dataset<Row> passengers = TitanicUtils.readPassengers(spark);

        // Step - 1: Make Vectors from dataframe's columns using special Vector Assmebler
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"pclass", "sibsp", "parch"})
                .setOutputCol("features");

        // Step - 2: Transform dataframe to vectorized dataframe with dropping rows
        Dataset<Row> output = assembler.transform(
            passengers.na().drop(new String[]{"pclass", "sibsp", "parch"}) // <============== drop row if it has nulls/NaNs in the next list of columns
        ).select("features", "survived");

        // Step - 3: Set up the Decision Tree Classifier
        DecisionTreeClassifier trainer = new DecisionTreeClassifier()
                .setLabelCol("survived")
                .setFeaturesCol("features");

        // Step - 4: Train the model
        DecisionTreeClassificationModel model = trainer.fit(output);

        // Step - 5: Predict with the model
        Dataset<Row> rawPredictions = model.transform(output);

        // Step - 6: Evaluate prediction
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("survived")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");

        // Step - 7: Calculate accuracy
        double accuracy = evaluator.evaluate(rawPredictions);
        System.out.println("Test Error = " + (1.0 - accuracy));

        // Step - 8: Print out the model
        System.out.println("Learned classification tree model:\n" + model.toDebugString());
    }
}
