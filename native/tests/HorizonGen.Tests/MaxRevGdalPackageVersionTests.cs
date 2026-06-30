using System.Xml.Linq;

namespace HorizonGen.Tests;

[TestClass]
public class MaxRevGdalPackageVersionTests
{
    [TestMethod]
    public void MaxRevGdalPackagesStayOnAdr0061PinnedVersion()
    {
        var repoRoot = FindRepoRoot();
        var projectFiles = Directory
            .EnumerateFiles(Path.Combine(repoRoot, "native", "new_horizon"), "*.csproj", SearchOption.AllDirectories)
            .ToArray();

        Assert.IsTrue(projectFiles.Length > 0, "Expected to find native .csproj files.");

        var mismatches = new List<string>();
        foreach (var projectFile in projectFiles)
        {
            var document = XDocument.Load(projectFile);
            foreach (var reference in document.Descendants("PackageReference"))
            {
                var include = reference.Attribute("Include")?.Value;
                if (include is not ("MaxRev.Gdal.Core" or "MaxRev.Gdal.LinuxRuntime.Minimal"))
                    continue;

                var version = reference.Attribute("Version")?.Value;
                if (version != "3.12.1.470")
                    mismatches.Add($"{Path.GetRelativePath(repoRoot, projectFile)} {include}={version}");
            }
        }

        CollectionAssert.AreEqual(Array.Empty<string>(), mismatches);
    }

    private static string FindRepoRoot()
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory is not null)
        {
            if (File.Exists(Path.Combine(directory.FullName, "requirements.txt"))
                && Directory.Exists(Path.Combine(directory.FullName, "native", "new_horizon")))
            {
                return directory.FullName;
            }
            directory = directory.Parent;
        }

        throw new InvalidOperationException("Could not locate repository root.");
    }
}
