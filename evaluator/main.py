"""Main CLI application for RAGAs-based RAG evaluation."""

import asyncio
import logging
import sys
from pathlib import Path
from typing import List, Optional

import click
import structlog
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.logging import RichHandler
from rich.table import Table
from rich.panel import Panel

from config import config
from ragas_evaluator import RAGASEvaluator
from dataset_generator import TestDatasetGenerator

# Configure structured logging
logging.basicConfig(
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
    level=getattr(logging, config.log_level.upper(), logging.INFO),
)

logger = structlog.get_logger()
console = Console()


@click.group()
def cli():
    """RAGAs Evaluator for Agentic RAG System."""
    pass


@cli.command()
@click.option('--num-questions', '-n', default=20, help='Number of test questions to generate')
@click.option('--output', '-o', help='Output path for the test dataset')
def generate_dataset(num_questions: int, output: Optional[str]):
    """Generate test dataset from indexed documents."""
    async def run_generation():
        generator = TestDatasetGenerator()

        try:
            console.print(" Initializing test dataset generator...", style="bold blue")

            if not await generator.initialize():
                console.print(" Failed to initialize dataset generator", style="red")
                sys.exit(1)

            console.print(" Dataset generator initialized", style="green")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task(
                    f"Generating {num_questions} test questions...",
                    total=None
                )

                questions = await generator.generate_questions_from_documents(num_questions)

                progress.update(task, completed=1)

            if not questions:
                console.print(" No questions generated", style="yellow")
                return

            # Save dataset
            await generator.save_dataset(questions, output)

            # Show summary
            console.print(f" Generated {len(questions)} test questions", style="bold green")

            # Display sample questions
            if questions:
                table = Table(title="Sample Generated Questions")
                table.add_column("Type", style="cyan")
                table.add_column("Question", style="white", max_width=80)
                table.add_column("Source", style="dim")

                for i, q in enumerate(questions[:5]):
                    table.add_row(
                        q.get('question_type', 'unknown'),
                        q['question'][:100] + "..." if len(q['question']) > 100 else q['question'],
                        q['source_document']
                    )

                console.print(table)

        except KeyboardInterrupt:
            console.print(" Dataset generation interrupted", style="red")
        except Exception as e:
            logger.error("Dataset generation failed", error=str(e))
            console.print(f" Dataset generation failed: {e}", style="red")
            sys.exit(1)
        finally:
            await generator.cleanup()

    asyncio.run(run_generation())


@cli.command()
@click.option('--dataset', '-d', help='Path to test dataset JSON file')
@click.option('--metrics', '-m', multiple=True,
              help='Specific metrics to evaluate (can be used multiple times)')
@click.option('--output', '-o', help='Output path for evaluation report')
@click.option('--questions', '-q', multiple=True,
              help='Custom questions to evaluate (can be used multiple times)')
def evaluate(dataset: Optional[str], metrics: tuple, output: Optional[str], questions: tuple):
    """Run RAGAs evaluation on the RAG system."""
    async def run_evaluation():
        evaluator = RAGASEvaluator()

        try:
            console.print(" Starting RAGAs evaluation...", style="bold blue")

            if not await evaluator.initialize():
                console.print(" Failed to initialize evaluator", style="red")
                sys.exit(1)

            console.print(" RAGAs evaluator initialized", style="green")

            # Convert metrics tuple to list if provided
            selected_metrics = list(metrics) if metrics else None

            # Convert questions tuple to list if provided
            custom_questions = list(questions) if questions else None

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Running complete evaluation...", total=None)

                report_path = await evaluator.run_complete_evaluation(
                    custom_questions=custom_questions,
                    metrics=selected_metrics
                )

                progress.update(task, completed=1)

            if report_path:
                console.print(f" Evaluation completed successfully!", style="bold green")
                console.print(f" Report saved to: {report_path}", style="cyan")

                # Try to display summary
                await display_evaluation_summary(report_path)
            else:
                console.print(" Evaluation failed", style="red")
                sys.exit(1)

        except KeyboardInterrupt:
            console.print(" Evaluation interrupted", style="red")
        except Exception as e:
            logger.error("Evaluation failed", error=str(e))
            console.print(f" Evaluation failed: {e}", style="red")
            sys.exit(1)
        finally:
            await evaluator.cleanup()

    asyncio.run(run_evaluation())


@cli.command()
def status():
    """Check the status of the evaluation system."""
    async def check_status():
        evaluator = RAGASEvaluator()

        try:
            console.print(" Checking system status...", style="bold blue")

            if not await evaluator.initialize():
                console.print(" System initialization failed", style="red")
                sys.exit(1)

            # Check database connection and stats
            doc_count = await evaluator.db_client.get_document_count()
            chunk_count = await evaluator.db_client.get_chunk_count()
            source_files = await evaluator.db_client.get_all_source_files()

            # Check RAG backend health
            backend_healthy = await evaluator.rag_client.health_check()

            # Display status
            status_panel = Panel.fit(
                f"""
Database Status:  Connected
Documents Indexed: {doc_count}
Total Chunks: {chunk_count}
Source Files: {len(source_files)}
Backend API: {' Healthy' if backend_healthy else ' Unavailable'}
                """.strip(),
                title="System Status",
                border_style="green" if backend_healthy else "yellow"
            )

            console.print(status_panel)

            # Show source files
            if source_files:
                table = Table(title="Indexed Documents")
                table.add_column("Document", style="cyan")

                for file in source_files:
                    table.add_row(file)

                console.print(table)

            # Check if test dataset exists
            dataset_path = Path(config.test_dataset_path)
            if dataset_path.exists():
                console.print(f" Test dataset found: {dataset_path}", style="green")
            else:
                console.print(f" No test dataset found. Run 'generate-dataset' first.", style="yellow")

        except Exception as e:
            logger.error("Status check failed", error=str(e))
            console.print(f" Status check failed: {e}", style="red")
            sys.exit(1)
        finally:
            await evaluator.cleanup()

    asyncio.run(check_status())


@cli.command()
@click.option('--report', '-r', required=True, help='Path to evaluation report JSON file')
def show_report(report: str):
    """Display evaluation report in a formatted way."""
    asyncio.run(display_evaluation_summary(report))


async def display_evaluation_summary(report_path: str):
    """Display evaluation report summary."""
    try:
        import json

        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)

        # Display evaluation metadata
        metadata = report.get('evaluation_metadata', {})
        console.print(f"\n Evaluation Report - {metadata.get('timestamp', 'Unknown')}", style="bold")

        # Display scores
        results = report.get('evaluation_results', {})
        scores = results.get('scores', {})

        if scores:
            table = Table(title="RAGAs Evaluation Scores")
            table.add_column("Metric", style="cyan")
            table.add_column("Score", justify="right", style="green")
            table.add_column("Std Dev", justify="right", style="dim")
            table.add_column("Range", justify="right", style="dim")

            for metric, score_data in scores.items():
                if isinstance(score_data, dict):
                    mean_score = f"{score_data.get('mean', 0):.3f}"
                    std_score = f"{score_data.get('std', 0):.3f}"
                    min_score = score_data.get('min', 0)
                    max_score = score_data.get('max', 0)
                    score_range = f"{min_score:.2f} - {max_score:.2f}"
                else:
                    mean_score = f"{float(score_data):.3f}"
                    std_score = "N/A"
                    score_range = "N/A"

                table.add_row(metric, mean_score, std_score, score_range)

            console.print(table)

        # Display summary
        summary = report.get('summary', {})
        if summary:
            summary_panel = Panel.fit(
                f"""
Overall Performance: {summary.get('overall_performance', 'Unknown')}
Best Metric: {summary.get('best_metric', 'N/A')}
Worst Metric: {summary.get('worst_metric', 'N/A')}
                """.strip(),
                title="Summary",
                border_style="blue"
            )
            console.print(summary_panel)

            # Display recommendations
            recommendations = summary.get('recommendations', [])
            if recommendations:
                console.print("\n Recommendations:", style="bold yellow")
                for i, rec in enumerate(recommendations, 1):
                    console.print(f"  {i}. {rec}")

        # Display system info
        db_stats = report.get('database_stats', {})
        if db_stats:
            console.print(f"\n📈 Database: {db_stats.get('document_count', 0)} docs, "
                         f"{db_stats.get('chunk_count', 0)} chunks", style="dim")

    except Exception as e:
        logger.error("Failed to display report", error=str(e))
        console.print(f" Failed to display report: {e}", style="red")


@cli.command()
@click.option('--path', '-p', help='Path to reports directory')
def list_reports(path: Optional[str]):
    """List available evaluation reports."""
    try:
        reports_dir = Path(path or config.reports_dir)

        if not reports_dir.exists():
            console.print(" No reports directory found", style="yellow")
            return

        report_files = list(reports_dir.glob("rag_evaluation_report_*.json"))

        if not report_files:
            console.print(" No evaluation reports found", style="yellow")
            return

        table = Table(title="Available Evaluation Reports")
        table.add_column("Report File", style="cyan")
        table.add_column("Created", style="dim")
        table.add_column("Size", style="dim", justify="right")

        for report_file in sorted(report_files, key=lambda x: x.stat().st_mtime, reverse=True):
            import datetime

            created_time = datetime.datetime.fromtimestamp(report_file.stat().st_mtime)
            file_size = report_file.stat().st_size

            size_str = f"{file_size / 1024:.1f} KB" if file_size > 1024 else f"{file_size} B"

            table.add_row(
                report_file.name,
                created_time.strftime("%Y-%m-%d %H:%M:%S"),
                size_str
            )

        console.print(table)
        console.print(f"\n Use 'show-report -r {reports_dir}/[filename]' to view a specific report")

    except Exception as e:
        logger.error("Failed to list reports", error=str(e))
        console.print(f" Failed to list reports: {e}", style="red")


if __name__ == '__main__':
    cli()