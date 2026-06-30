using System.Threading.Tasks.Dataflow;

namespace moonlib.pipeline
{
    public class Pipeline<T>
    {
        private readonly List<IDataflowBlock> _blocks = new List<IDataflowBlock>();
        private ITargetBlock<T>? _firstBlock;
        private IDataflowBlock? _lastBlock;

        public void AddStep(
            Func<T, Task<T>> stepFunc,
            int maxDegreeOfParallelism = 4,
            int boundedCapacity = 10,
            bool ensureOrdered = true)
        {
            var options = new ExecutionDataflowBlockOptions
            {
                MaxDegreeOfParallelism = maxDegreeOfParallelism,
                BoundedCapacity = boundedCapacity,
                EnsureOrdered = ensureOrdered
            };

            var block = new TransformBlock<T, T>(async input =>
            {
                try
                {
                    return await stepFunc(input);
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"Error processing item: {ex.Message}");
                    throw;
                }
            }, options);

            if (_blocks.Count == 0)
            {
                _firstBlock = block as ITargetBlock<T>;
            }
            else
            {
                var previousBlock = _blocks[_blocks.Count - 1] as ISourceBlock<T>;
                if (previousBlock != null)
                {
                    previousBlock.LinkTo(block, new DataflowLinkOptions { PropagateCompletion = true });
                }
                else
                {
                    throw new InvalidOperationException("Previous block does not implement ISourceBlock<TCurrentInput>.");
                }
            }

            _blocks.Add(block);
            _lastBlock = block;
        }

        public void AddTerminalStep(
            Func<T, Task> stepFunc,
            int maxDegreeOfParallelism = 4,
            int boundedCapacity = 10,
            bool ensureOrdered = true)
        {
            var options = new ExecutionDataflowBlockOptions
            {
                MaxDegreeOfParallelism = maxDegreeOfParallelism,
                BoundedCapacity = boundedCapacity,
                EnsureOrdered = ensureOrdered
            };

            var block = new ActionBlock<T>(async input =>
            {
                try
                {
                    await stepFunc(input);
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"Error processing item: {ex.Message}");
                    throw;
                }
            }, options);

            if (_blocks.Count == 0)
            {
                _firstBlock = block as ITargetBlock<T>;
            }
            else
            {
                var previousBlock = _blocks[_blocks.Count - 1] as ISourceBlock<T>;
                if (previousBlock != null)
                {
                    previousBlock.LinkTo(block, new DataflowLinkOptions { PropagateCompletion = true });
                }
                else
                {
                    throw new InvalidOperationException("Previous block does not implement ISourceBlock<TCurrentInput>.");
                }
            }

            _blocks.Add(block);
            _lastBlock = block;
        }

        public async Task ProcessAsync(IEnumerable<T> inputs)
        {
            if (_firstBlock == null || _lastBlock == null)
                throw new InvalidOperationException("Pipeline has no processing steps.");

            foreach (var input in inputs)
            {
                await _firstBlock.SendAsync(input);
            }

            _firstBlock.Complete();
            await _lastBlock.Completion;
        }
    }
}