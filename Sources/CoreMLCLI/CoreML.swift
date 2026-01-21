import ArgumentParser

@main
struct CoreML: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "coreml",
        abstract: "A command-line tool for working with Core ML models",
        version: "1.0.0",
        subcommands: [
            Inspect.self,
            Predict.self,
            Batch.self,
            Benchmark.self,
            Compile.self,
            Meta.self
        ],
        defaultSubcommand: nil
    )
}
