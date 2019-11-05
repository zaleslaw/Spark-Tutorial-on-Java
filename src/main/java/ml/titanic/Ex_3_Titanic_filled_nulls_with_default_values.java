package ml.titanic;

import java.util.HashMap;
import java.util.Map;
import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

/**
 * Fill the data with the default values presented in Map object
 */
public class Ex_3_Titanic_filled_nulls_with_default_values {
    public static void main(String[] args) {
        SparkSession spark = TitanicUtils.getSparkSession();

        Dataset<Row> passengers = TitanicUtils.readPassengersWithCastingToDoubles(spark);

        // Step - 1: Make Vectors from dataframe's columns using special Vector Assmebler
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"pclass", "sibsp", "parch"})
                .setOutputCol("features");

        // Step - 2: Define default values for missing data
        Map<String, Object> replacements = new HashMap<>();
        replacements.put("pclass", 1);
        replacements.put("sibsp", 0);
        replacements.put("parch", 0);

        // Step - 3: Fill the data with the default values
        Dataset<Row> passengersWithFilledEmptyValues = passengers.na().fill(replacements);

        passengersWithFilledEmptyValues.show(); // <= check first row

        // Step - 4: Transform dataframe to vectorized dataframe with dropping rows
        Dataset<Row> output = assembler.transform(
            passengersWithFilledEmptyValues
        ).select("features", "survived");

        // Step - 5: Set up the Decision Tree Classifier
        DecisionTreeClassifier trainer = new DecisionTreeClassifier()
                .setLabelCol("survived")
                .setFeaturesCol("features");

        // Step - 6: Train the model
        DecisionTreeClassificationModel model = trainer.fit(output);

        // Step - 7: Predict with the model
        Dataset<Row> rawPredictions = model.transform(output);

        // Step - 8: Evaluate prediction
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("survived")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");

        // Step - 9: Calculate accuracy
        double accuracy = evaluator.evaluate(rawPredictions);
        System.out.println("Test Error = " + (1.0 - accuracy));

        // Step - 10: Print out the model
        System.out.println("Learned classification tree model:\n" + model.toDebugString());
    }
}
