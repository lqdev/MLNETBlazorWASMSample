using System;
using System.Linq;
using Microsoft.ML;
using SchemaLibrary;

namespace TrainingConsole
{
    class Program
    {
        static void Main(string[] args)
        {
            // 1. Initialize MLContext
            MLContext mlContext = new MLContext();

            // 2. Load the data
            IDataView data = mlContext.Data.LoadFromTextFile<ModelInput>("iris.data", separatorChar:',');

            // 3. Shuffle the data
            IDataView shuffledData = mlContext.Data.ShuffleRows(data);

            // 3. Define the data preparation and training pipeline.
            IEstimator<ITransformer> pipeline = 
                mlContext.Transforms.Concatenate("Features","SepalLength","SepalWidth","PetalLength","PetalWidth")
                    .Append(mlContext.Transforms.NormalizeMinMax("Features"))
                    .Append(mlContext.Transforms.Conversion.MapValueToKey("Label"))
                    .Append(mlContext.MulticlassClassification.Trainers.NaiveBayes())
                    .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            // 4. Train with cross-validation
            var cvResults = mlContext.MulticlassClassification.CrossValidate(shuffledData, pipeline);

            // 5. Get the highest performing model and its accuracy
            (ITransformer, double) model = 
                cvResults
                    .OrderByDescending(fold => fold.Metrics.MacroAccuracy)
                    .Select(fold => (fold.Model, fold.Metrics.MacroAccuracy))
                    .First();

            Console.WriteLine($"Top performing model's macro-accuracy: {model.Item2}");

            // 6. Save the model
            mlContext.Model.Save(model.Item1, data.Schema, "model.zip");

            Console.WriteLine("Model trained");
        }
    }
}
