import Foundation
import XCTest

/// Shared helper for loading test resources from TestResources directory
enum TestResourceLoader {

    // MARK: - Vocadito Dataset

    /// Get path to Vocadito audio file in TestResources
    /// - Parameter filename: Audio filename (e.g., "vocadito_1.wav")
    /// - Returns: Full path to the audio file
    static func getVocaditoAudioPath(filename: String) -> String {
        // Try without subdirectory first (files are in bundle root)
        let testBundle = Bundle(for: DummyTestClass.self)
        let fileURL = URL(fileURLWithPath: filename)
        let name = fileURL.deletingPathExtension().lastPathComponent
        let ext = fileURL.pathExtension

        if let url = testBundle.url(forResource: name, withExtension: ext) {
            return url.path
        }

        // Fallback to subdirectory structure
        return getVocaditoResourcePath(filename: filename, subdirectory: "Audio")
    }

    /// Get path to Vocadito F0 annotation file in TestResources
    /// - Parameter filename: F0 annotation filename (e.g., "vocadito_1_f0.csv")
    /// - Returns: Full path to the F0 annotation file
    static func getVocaditoF0Path(filename: String) -> String {
        return getVocaditoResourcePath(filename: filename, subdirectory: "Annotations/F0")
    }

    /// Get path to Vocadito note annotation file in TestResources
    /// - Parameter filename: Note annotation filename (e.g., "vocadito_1_notesA1.csv")
    /// - Returns: Full path to the note annotation file
    static func getVocaditoNotePath(filename: String) -> String {
        return getVocaditoResourcePath(filename: filename, subdirectory: "Annotations/Notes")
    }

    /// Get path to TestNotes.json file
    /// - Returns: Full path to the TestNotes.json file
    static func getVocaditoTestNotesPath() -> String {
        // Use Bundle to find TestNotes.json
        let testBundle = Bundle(for: DummyTestClass.self)

        // Try without subdirectory first (file is in bundle root)
        if let url = testBundle.url(forResource: "TestNotes", withExtension: "json") {
            return url.path
        }

        // Fallback to subdirectory structure
        guard let url = testBundle.url(forResource: "TestNotes", withExtension: "json", subdirectory: "Vocadito") else {
            fatalError("TestNotes.json not found in test bundle. Make sure it's added to VocalMasteringLabTests target.")
        }
        return url.path
    }

    // MARK: - Private Helpers

    /// Get path to Vocadito resource file
    /// - Parameters:
    ///   - filename: Resource filename
    ///   - subdirectory: Subdirectory within Vocadito test resources (e.g., "Audio", "Annotations/F0")
    /// - Returns: Full path to the resource file
    private static func getVocaditoResourcePath(filename: String, subdirectory: String) -> String {
        // Use Bundle to find resource files
        let testBundle = Bundle(for: DummyTestClass.self)

        // Extract file name without extension
        let fileURL = URL(fileURLWithPath: filename)
        let name = fileURL.deletingPathExtension().lastPathComponent
        let ext = fileURL.pathExtension

        // Construct subdirectory path within Vocadito
        let fullSubdirectory = "Vocadito/\(subdirectory)"

        guard let url = testBundle.url(forResource: name, withExtension: ext, subdirectory: fullSubdirectory) else {
            fatalError("\(filename) not found in test bundle at \(fullSubdirectory). Make sure it's added to VocalMasteringLabTests target.")
        }
        return url.path
    }
}

// MARK: - Dummy Class for Bundle

/// Dummy class for accessing test bundle
private class DummyTestClass: XCTestCase {}
