import Foundation

/// Word Error Rate-based accuracy for ASR evaluation.
enum AccuracyEngine {

    /// Normalize to lowercase words, stripping punctuation.
    static func words(_ text: String) -> [String] {
        text.lowercased()
            .unicodeScalars
            .map { CharacterSet.letters.union(.decimalDigits).contains($0) ? Character($0) : " " }
            .map(String.init)
            .joined()
            .split(separator: " ")
            .map(String.init)
            .filter { !$0.isEmpty }
    }

    /// Word-level Levenshtein edit distance (substitutions, deletions, insertions).
    static func editDistance(_ a: [String], _ b: [String]) -> Int {
        let m = a.count, n = b.count
        if m == 0 { return n }
        if n == 0 { return m }
        var dp = Array(repeating: Array(repeating: 0, count: n + 1), count: m + 1)
        for i in 0...m { dp[i][0] = i }
        for j in 0...n { dp[0][j] = j }
        for i in 1...m {
            for j in 1...n {
                dp[i][j] = a[i-1] == b[j-1]
                    ? dp[i-1][j-1]
                    : 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
            }
        }
        return dp[m][n]
    }

    /// Returns accuracy in 0–1 (= 1 − WER, clamped to ≥ 0).
    /// Returns nil when the reference is empty.
    static func accuracy(hypothesis: String, reference: String) -> Double? {
        let ref = words(reference)
        guard !ref.isEmpty else { return nil }
        let hyp = words(hypothesis)
        let wer = Double(editDistance(hyp, ref)) / Double(ref.count)
        return max(0, 1 - wer)
    }
}
