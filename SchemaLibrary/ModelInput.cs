using Microsoft.ML.Data;

namespace SchemaLibrary
{
    public class ModelInput
    {
        [LoadColumn(0)]
        public float SepalLength { get; set; }
        
        [LoadColumn(1)]
        public float SepalWidth { get; set; }
        
        [LoadColumn(2)]
        public float PetalLength { get; set; }
        
        [LoadColumn(3)]
        public float PetalWidth { get; set; }
        
        [LoadColumn(4)]
        public string Label { get; set; }
    }
}
