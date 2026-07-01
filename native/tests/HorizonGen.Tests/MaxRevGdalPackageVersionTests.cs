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
            .EnumerateFiles(Path.Combine(repoRoot, "native"), "*.csproj", SearchOption.AllDirectories)
            .Where(path => !path.Split(Path.DirectorySeparatorChar).Contains("bin")
                && !path.Split(Path.DirectorySeparatorChar).Contains("obj"))
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
            if (File.Exists(Path.Combine(directory.FullName, "pyproject.toml"))
                && Directory.Exists(Path.Combine(directory.FullName, "native", "moonlib")))
            {
                return directory.FullName;
            }
            directory = directory.Parent;
        }

        throw new InvalidOperationException("Could not locate repository root.");
    }
}
