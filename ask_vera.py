import argparse
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from vera_agent import VeraAgent

# ANSI escape codes for coloring
ORANGE = '\033[38;5;208m'
YELLOW = '\033[93m'
CYAN = '\033[36m'
GREEN = '\033[92m'
RESET = '\033[0m'

def main():
    """
    Main function to parse command line arguments and run the VeraAgent.
    """
    parser = argparse.ArgumentParser(
        description=f"{GREEN}Run the VeraAgent to generate profound insights using the VERA Protocol.{RESET}"
    )

    parser.add_argument(
        "query",
        type=str,
        nargs='?', # Makes the query optional
        help="The open-ended question or problem to analyze."
    )
    
    parser.add_argument(
        "-f", "--file",
        type=str,
        help="Path to a file containing the prompt. This overrides the direct query."
    )

    parser.add_argument(
        "-c", "--context",
        type=str,
        nargs='+',  # Accepts one or more file paths for context
        help="One or more paths to files containing context documents."
    )

    parser.add_argument(
        "-m", "--model",
        type=str,
        default=None,
        help="Optional: Override the LLM model specified in vera.yaml."
    )

    parser.add_argument(
        "-d", "--domains",
        type=str,
        default="1",
        help="""
        Number of domains to randomly select (e.g., '3'), or a semicolon-separated string of specific domains (e.g., 'parenting;coaching soccer').
        Default is '1'.
        """
    )

    args = parser.parse_args()
    
    # Handle the query input from file or command line
    if args.file:
        try:
            with open(args.file, 'r') as f:
                original_query = f.read()
        except FileNotFoundError:
            print(f"{GREEN}System: {RESET}Error: The file '{args.file}' was not found.")
            sys.exit(1)
    elif args.query:
        original_query = args.query
    else:
        print(f"{GREEN}System: {RESET}Error: No query or file was provided. Please provide a query string or a file path.")
        sys.exit(1)

    # Read context files if provided
    context_files_content = []
    if args.context:
        for file_path in args.context:
            try:
                print(f"{GREEN}System: {RESET}Opening {file_path}")
                with open(file_path, 'r') as f:
                    context_files_content.append(f.read())
            except FileNotFoundError:
                print(f"{GREEN}System: {RESET}Error: Context file '{file_path}' was not found.")
                sys.exit(1)

    # Determine input types and convert as needed
    try:
        num_domains = int(args.domains)
        domains = None
    except ValueError:
        domains = args.domains.split(';')
        num_domains = len(domains)
    
    # Initialize the VeraAgent, passing the model if provided
    agent = VeraAgent(model_name=args.model)
    
    # Get the instruction prompt and original query and print them
    instruction_prompt = agent.get_instruction_prompt(domain="<dummy_domain_for_print>")
    print(f"\n{GREEN}--- Instruction Prompt ---{RESET}")
    print(f"{ORANGE}{instruction_prompt}{RESET}")
    print(f"\n{GREEN}--- Original Query ---{RESET}")
    print(f"{ORANGE}{original_query}{RESET}")

    # Now, intelligently insert the instruction into the context
    full_context_for_llm = ""
    if context_files_content:
        full_context_for_llm = agent.get_intelligent_context(
            instruction_prompt,
            original_query,
            context_files_content
        )
        print(f"\n{GREEN}--- Original Context ---{RESET}")
        print(f"{CYAN}{context_files_content}{RESET}")
        print(f"\n{GREEN}--- Final Context Sent to LLM (with instructions intelligently placed) ---{RESET}")
        print(f"{CYAN}{full_context_for_llm}{RESET}\n\n")

    # First, get the domains from the agent
    print(f"\n{GREEN}System: {RESET}Constructing VERA Protocol based on input parameters.")
    domains_used = agent.get_domains(
        domains_input=domains or num_domains
    )
    
    # Print the generated domains for the user to see before processing
    print(f"{GREEN}Generated Domains:{RESET} {YELLOW}{', '.join(domains_used)}{RESET}")
    print(f"{GREEN}Generated Perspectives:{RESET} {YELLOW}{'None'}{RESET}")
    print(f"{GREEN}System: {RESET}Beginning parallel iterative analysis.{RESET}")

    individual_responses = []

    with ThreadPoolExecutor(max_workers=os.cpu_count() or 1) as executor:
        future_to_domain = {
            executor.submit(agent.process_domain, original_query, domain, full_context_for_llm): domain 
            for domain in domains_used
        }
        for future in as_completed(future_to_domain):
            domain = future_to_domain[future]
            try:
                response = future.result()
                individual_responses.append(response)
                print(f"\n{YELLOW}Fragment from domain '{domain}':{RESET}\n{response}\n")
            except Exception as exc:
                print(f'{GREEN}System: {RESET}Domain {domain} generated an exception: {exc}')

    # Now, run the synthesis
    print(f"\n{GREEN}System: {RESET}Synthesizing all individual responses into a final, comprehensive answer.")
    final_wisdom, synthesis_prompt = agent.synthesize_wisdom(
        original_query=original_query,
        individual_responses=individual_responses
    )
    
    # Print the full report as before
    print(f"\n{GREEN}--- VERA Protocol Report ---{RESET}")
    print(f"{GREEN}Original Query:{RESET} {ORANGE}{original_query}{RESET}")
    print(f"{GREEN}Generated Domains:{RESET} {YELLOW}{', '.join(domains_used)}{RESET}")
    print(f"{GREEN}Generated Perspectives:{RESET} {YELLOW}{'None'}{RESET}")
    
    # Print individual responses
    print(f"\n{GREEN}--- All Individual Wisdom Fragments (before synthesis) ---{RESET}")
    for i, resp in enumerate(individual_responses):
        print(f"\n{YELLOW}Fragment {i+1}:{RESET}\n{resp}\n")

    # Print the final wisdom
    print(f"\n{GREEN}--- The Final VERA Protocol Wisdom ---{RESET}")
    print(f"{CYAN}{final_wisdom}{RESET}")

if __name__ == "__main__":
    main()

