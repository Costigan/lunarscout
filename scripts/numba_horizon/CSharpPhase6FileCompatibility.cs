using System.Security.Cryptography;
using System.Text.Json;
using moonlib.horizon;

if (args.Length != 2)
{
    Console.Error.WriteLine(
        "Usage: CSharpPhase6FileCompatibility <python-horizon-input> <csharp-roundtrip-output>");
    return 2;
}

string inputPath = Path.GetFullPath(args[0]);
string outputPath = Path.GetFullPath(args[1]);
float[] values = HorizonFile.ReadHorizonFile(inputPath);
HorizonFile.WriteHorizonFile(outputPath, values);

byte[] valueBytes = new byte[values.Length * sizeof(float)];
Buffer.BlockCopy(values, 0, valueBytes, 0, valueBytes.Length);
Console.WriteLine(JsonSerializer.Serialize(new
{
    input_path = inputPath,
    output_path = outputPath,
    sample_count = values.Length,
    value_sha256 = Convert.ToHexString(SHA256.HashData(valueBytes)).ToLowerInvariant(),
    selected_values = new[] { values[0], values[1439], values[1440], values[^1] },
    output_bytes = new FileInfo(outputPath).Length,
}, new JsonSerializerOptions { WriteIndented = true }));
return 0;
