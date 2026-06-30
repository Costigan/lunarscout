namespace moonlib.pipeline
{
    public class PipelineStep<TInput, TOutput>
    {
        public Func<TInput, Task<TOutput>> Func { get; }

        public PipelineStep(Func<TInput, TOutput> syncFunc)
        {
            Func = input => Task.FromResult(syncFunc(input));
        }

        public PipelineStep(Func<TInput, Task<TOutput>> asyncFunc)
        {
            Func = asyncFunc;
        }
    }
}
