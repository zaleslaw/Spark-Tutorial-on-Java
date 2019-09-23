package ml.titanic;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.Imputer;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

/**
 * The same result with Pipeline API. Accuracy = 0.224
 * <p>
 * Q: Why do we have two DropSex is working prints?
 * <p>
 * A: One from trainer and one from model.
 */
public class Ex_7_Titanic_refactor_to_pipeline {

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

        // Step - 2: Define the Indexer for the column "embarked"
        StringIndexer embarkedIndexer = new StringIndexer()
            .setInputCol("embarked")
            .setOutputCol("embarkedIndexed")
            .setHandleInvalid("keep"); // special mode to create special double value for null values

        // Step - 3: Define strategy and new column names for Imputer transformation
        Imputer imputer = new Imputer()
            .setInputCols(new String[]{"pclass", "sibsp", "parch"})
            .setOutputCols(new String[]{"pclass_imputed", "sibsp_imputed", "parch_imputed"})
            .setStrategy("mean");

        // Step - 4: Make Vectors from dataframe's columns using special Vector Assembler
        VectorAssembler assembler = new VectorAssembler()
            .setInputCols(new String[]{"pclass_imputed", "sibsp_imputed", "parch_imputed", "sexIndexed", "embarkedIndexed"})
            .setOutputCol("features");

        // Step - 5: Set up the Decision Tree Classifier
        DecisionTreeClassifier trainer = new DecisionTreeClassifier()
                .setLabelCol("survived")
                .setFeaturesCol("features");

        // Step - 6: Chain all stages in one Pipeline
        Pipeline pipeline = new Pipeline()
            .setStages(new PipelineStage[]{sexIndexer, embarkedIndexer, new TitanicUtils.DropSex(), imputer, assembler, trainer});

        // Step - 7: Train the model
        PipelineModel model = pipeline.fit(training);

        // Step - 8: Predict with the model
        Dataset<Row> rawPredictions = model.transform(test);

        // Step - 9: Evaluate prediction
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("survived")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");

        // Step - 10: Calculate accuracy
        double accuracy = evaluator.evaluate(rawPredictions);
        System.out.println("Test Error = " + (1.0 - accuracy));
    }
}
