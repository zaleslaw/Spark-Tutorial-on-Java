package ml.titanic;

import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.Imputer;
import org.apache.spark.ml.feature.ImputerModel;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

/**
 * Add two features: "sex" and "embarked". They are presented as sets of string. Accuracy = 0.1913
 * <p>
 * Old columns should be dropped from the dataset to use imputer
 * <p>
 * The first row in imputed dataset is filled with the special values
 */
public class Ex_6_Titanic_use_categorial_feature {
    public static void main(String[] args) {
        SparkSession spark = TitanicUtils.getSparkSession();

        Dataset<Row> passengers = TitanicUtils.readPassengersWithCastingToDoubles(spark)
            .select("survived", "pclass", "sibsp", "parch", "sex", "embarked");

        Dataset<Row>[] split = passengers.randomSplit(new double[] {0.7, 0.3}, 12345);
        Dataset<Row> training = split[0].cache();
        Dataset<Row> test = split[1].cache();

        // Step - 1: Define the Indexer for the column "sex"
        StringIndexer sexIndexer = new StringIndexer()
            .setInputCol("sex")
            .setOutputCol("sexIndexed")
            .setHandleInvalid("keep"); // special mode to create special double value for null values

        StringIndexerModel simodelSex = sexIndexer.fit(training);

        Dataset<Row> passengersWithIndexedSex = simodelSex.transform(training);

        // Step - 2: Define the Indexer for the column "embarked"
        StringIndexer embarkedIndexer = new StringIndexer()
            .setInputCol("embarked")
            .setOutputCol("embarkedIndexed")
            .setHandleInvalid("keep"); // special mode to create special double value for null values

        StringIndexerModel simodelForEmbarked = embarkedIndexer
            .fit(passengersWithIndexedSex);

        Dataset<Row> passengersWithIndexedCategorialFeatures = simodelForEmbarked
            .transform(passengersWithIndexedSex)
            .drop("sex", "embarked"); // <============== drop columns to use Imputer

        passengersWithIndexedCategorialFeatures.show();
        passengersWithIndexedCategorialFeatures.printSchema();

        // Step - 3: Define strategy and new column names for Imputer transformation
        Imputer imputer = new Imputer()
            .setInputCols(new String[]{"pclass", "sibsp", "parch"})
            .setOutputCols(new String[]{"pclass_imputed", "sibsp_imputed", "parch_imputed"})
            .setStrategy("mean");

        // Step - 4: Transform the dataset with the Imputer
        ImputerModel imputerModel = imputer.fit(training);
        Dataset<Row> passengersWithFilledEmptyValues = imputerModel.transform(passengersWithIndexedCategorialFeatures);

        passengersWithFilledEmptyValues.show(); // <= check first row


        // Step - 5: Make Vectors from dataframe's columns using special Vector Assembler
        VectorAssembler assembler = new VectorAssembler()
            .setInputCols(new String[]{"pclass_imputed", "sibsp_imputed", "parch_imputed", "sexIndexed", "embarkedIndexed"})
            .setOutputCol("features");

        // Step - 6: Transform dataframe to vectorized dataframe with dropping rows
        Dataset<Row> output = assembler.transform(
            passengersWithFilledEmptyValues
        ).select("features", "survived");

        // Step - 7: Set up the Decision Tree Classifier
        DecisionTreeClassifier trainer = new DecisionTreeClassifier()
                .setLabelCol("survived")
                .setFeaturesCol("features");

        // Step - 8: Train the model
        DecisionTreeClassificationModel model = trainer.fit(output);

        // Step - 9: Predict with the model
        Dataset<Row> rawPredictions = model.transform(
                    assembler.transform
                        (imputerModel.transform(
                            simodelForEmbarked.transform(
                                simodelSex.transform(test)))));

        // Step - 10: Evaluate prediction
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("survived")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");

        // Step - 11: Calculate accuracy
        double accuracy = evaluator.evaluate(rawPredictions);
        System.out.println("Test Error = " + (1.0 - accuracy));

        // Step - 12: Print out the model
        System.out.println("Learned classification tree model:\n" + model.toDebugString());
    }
}
