using ILGPU.Runtime;
using moonlib.horizon;

namespace moonlib.tests
{
    [TestClass]
    [TestCategory("Integration")]
    [TestCategory("GpuBaseline")]
    public sealed class QuadTreeGpuBaselineTests
    {
        [TestMethod]
        public void Generator_SelectsCudaAccelerator()
        {
            if (Environment.GetEnvironmentVariable("LUNARSCOUT_REQUIRE_CUDA_BASELINE") != "1")
            {
                Assert.Inconclusive(
                    "Set LUNARSCOUT_REQUIRE_CUDA_BASELINE=1 to require the host CUDA probe.");
            }

            using var generator = new QuadTreeHorizonGenerator(disableHierarchy: false);

            Assert.AreEqual(
                AcceleratorType.Cuda,
                generator.SelectedAcceleratorType,
                $"Expected a CUDA accelerator, but selected {generator.SelectedAcceleratorName} " +
                $"({generator.SelectedAcceleratorType}).");
            Assert.IsFalse(string.IsNullOrWhiteSpace(generator.SelectedAcceleratorName));
            TestContext.WriteLine(
                $"Selected ILGPU accelerator: {generator.SelectedAcceleratorName} " +
                $"({generator.SelectedAcceleratorType})");
        }

        public TestContext TestContext { get; set; } = null!;
    }
}
