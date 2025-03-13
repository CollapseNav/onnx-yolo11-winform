using Microsoft.ML.OnnxRuntime.Tensors;
using onnx_yolo11_winform;
public static class YoloExt
{
    public static List<Prediction> GetPrediction(this Tensor<float> tensors, int[]? dimensions, float confidence)
    {
        List<Prediction> pres = new List<Prediction>();
        for (int i = 0; i < dimensions[2]; i++)
        {
            for (var j = 4; j < dimensions[1]; j++)
            {
                var value = tensors[0, j, i];
                if (value > confidence)
                {
                    var x = tensors[0, 0, i];
                    var y = tensors[0, 1, i];
                    var width = tensors[0, 2, i];
                    var height = tensors[0, 3, i];
                    float xmin = x - width / 2;
                    float ymin = y - height / 2;
                    float xmax = x + width / 2;
                    float ymax = y + height / 2;
                    pres.Add(new Prediction()
                    {
                        Label = Form1.Labels[j - 4],
                        Confidence = value,
                        Box = new Box(xmin, ymin, xmax, ymax)
                    });
                }
            }
        }
        return pres;
    }
    public static List<Prediction> NonMaximumSuppression(List<Prediction> predictions, float iouThreshold)
    {
        var combined = predictions.Zip(predictions.Select(i => i.Confidence).ToArray(), (prediction, score) => new { Box = prediction.Box.box, Score = score, prediction }).ToList();

        combined.Sort((a, b) => b.Score.CompareTo(a.Score));

        List<float[]> keepBoxes = new List<float[]>();
        List<Prediction> keepPredictions = new List<Prediction>();

        while (combined.Count > 0)
        {
            float[] bestBox = combined[0].Box;
            keepBoxes.Add(bestBox);
            keepPredictions.Add(combined[0].prediction);
            combined.RemoveAt(0);
            combined.RemoveAll(item => CalculateIoU(bestBox, item.Box) > iouThreshold);
        }

        return keepPredictions;
    }

    public static float GetAdjustedFontsize(List<Prediction> predictions)
    {
        float adjustedFontSize = 12;

        if (predictions.Count > 0)
        {
            int maxPredictionTextLength = predictions.Select(p => p.Label.Length).ToList().Max() + 5;
            float minPredictionBoxWidth = predictions.Select(p => p.Box!.Xmax - p.Box!.Xmin).ToList().Min();
            adjustedFontSize = Math.Clamp(minPredictionBoxWidth / ((float)maxPredictionTextLength), 8, 16);
        }

        return adjustedFontSize;
    }
    public static Bitmap RenderPredictions(this Bitmap image, List<Prediction> predictions)
    {
        using Graphics g = Graphics.FromImage(image);
        float markerSize = (image.Width + image.Height) * 0.001f;
        using Pen pen = new(Color.Red, markerSize);
        using Brush brush = new SolidBrush(Color.Blue);
        using Font font = new("Arial", GetAdjustedFontsize(predictions));
        predictions = NonMaximumSuppression(predictions, 0.5f);
        foreach (var p in predictions)
        {
            if (p == null || p.Box == null)
            {
                continue;
            }

            // Draw the box
            g.DrawLine(pen, p.Box.Xmin, p.Box.Ymin, p.Box.Xmax, p.Box.Ymin);
            g.DrawLine(pen, p.Box.Xmax, p.Box.Ymin, p.Box.Xmax, p.Box.Ymax);
            g.DrawLine(pen, p.Box.Xmax, p.Box.Ymax, p.Box.Xmin, p.Box.Ymax);
            g.DrawLine(pen, p.Box.Xmin, p.Box.Ymax, p.Box.Xmin, p.Box.Ymin);

            string labelText = $"{p.Label}, {p.Confidence:0.00}";
            g.DrawString(labelText, font, brush, new PointF(p.Box.Xmin, p.Box.Ymin));
        }
        return image;
    }

    public static float CalculateIoU(float[] box1, float[] box2)
    {
        float xminInter = Math.Max(box1[0], box2[0]);
        float yminInter = Math.Max(box1[1], box2[1]);
        float xmaxInter = Math.Min(box1[2], box2[2]);
        float ymaxInter = Math.Min(box1[3], box2[3]);

        float interArea = Math.Max(0, xmaxInter - xminInter) * Math.Max(0, ymaxInter - yminInter);
        float box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1]);
        float box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1]);
        float unionArea = box1Area + box2Area - interArea;

        return interArea / unionArea;
    }
}