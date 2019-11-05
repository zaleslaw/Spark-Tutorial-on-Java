package ml.titanic;

import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.Imputer;
import org.apache.spark.ml.feature.ImputerModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

/**
 * Fill missed values with average values
 *
 * But imputer needs in Double values in the whole dataset. Accuracy =  0.327
 */
public class Ex_5_Titanic_split_test_train_data {
    public static void main(String[] args) {
        SparkSession spark = TitanicUtils.getSparkSession();

        Dataset<Row> passengers = TitanicUtils.readPassengersWithCastingToDoubles(spark)
            .select("survived", "pclass", "sibsp", "parch");

        Dataset<Row>[] split = passengers.randomSplit(new double[] {0.7, 0.3}, 12345);
        Dataset<Row> training = split[0].cache();
        Dataset<Row> test = split[1].cache();

        // Step - 1: Make Vectors from dataframe's columns using special Vector Assmebler
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"pclass_imputed", "sibsp_imputed", "parch_imputed"})
                .setOutputCol("features");

        // Step - 2: Define strategy and new column names for Imputer transformation
        Imputer imputer = new Imputer()
            .setInputCols(new String[]{"pclass", "sibsp", "parch"})
            .setOutputCols(new String[]{"pclass_imputed", "sibsp_imputed", "parch_imputed"})
            .setStrategy("mean");

        // Step - 3: Transform the dataset with the Imputer
        ImputerModel imputerModel = imputer.fit(training);
        Dataset<Row> passengersWithFilledEmptyValues = imputerModel.transform(training);

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
        Dataset<Row> rawPredictions = model.transform(assembler.transform(imputerModel.transform(test)));

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
