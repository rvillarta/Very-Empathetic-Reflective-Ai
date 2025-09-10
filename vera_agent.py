import yaml
import requests
import json
import sys
import os
import random

# ANSI escape codes for coloring
ORANGE = '\033[38;5;208m'
YELLOW = '\033[93m'
CYAN = '\033[36m'
GREEN = '\033[92m'
RESET = '\033[0m'

class VeraAgent:
    """
    A Python-based agent that uses the VERA Protocol to generate profound insights.
    """

    def __init__(self, model_name=None):
        self.config = self._load_config()
        self.llm_model = model_name or self.config.get('llm_model', 'default-llm')
        self.domains_llm_model = self.config.get('domains_llm_model', self.llm_model)
        self.default_llm_temperature = self.config.get('default_llm_temperature', 0.8)
        self.domains_llm_temperature = self.config.get('domains_llm_temperature', 1.0)
        self.synthesis_llm_temperature = self.config.get('synthesis_llm_temperature', 0.5)
        self.base_url = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")

    def _load_config(self):
        """Loads the YAML configuration file."""
        try:
            with open('vera.yaml', 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"{GREEN}System: {RESET}Error: vera.yaml not found. Please ensure it's in the same directory.")
            sys.exit(1)

    def _call_ollama(self, prompt, model, temperature, context_data=None, json_mode=False):
        """Calls the local Ollama API to generate a response."""
        print(f"{GREEN}System: {RESET}Calling LLM...")
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "context": context_data,
            "options": {
                'seed': random.randint(0, 2**32 - 1),
                'temperature': temperature
            }

        }
        if json_mode:
            data["format"] = "json"
        try:
            response = requests.post(self.base_url, json=data)
            response.raise_for_status()
            result = response.json()
            return result['response'], result.get('context')
        except requests.exceptions.RequestException as e:
            return f"Error communicating with Ollama: {e}", None
        except json.JSONDecodeError:
            print(f"{GREEN}System: {RESET}Failed to decode JSON from LLM response.")
            return "Failed to parse JSON response.", None

    def _generate_list_from_llm(self, list_type, count, context):
        """Generates a list of domains or perspectives using the LLM and JSON format."""
        prompt = f"""
        You are a creative thinking assistant. Generate {count} diverse and unrelated {list_type}.
        Do not provide any extra text or conversational filler, just a JSON object.
        The JSON object should have a single key '{list_type}' which contains a list of strings.
        Example: {{"{list_type}": ["item1", "item2"]}}
        """
        response, _ = self._call_ollama(
            prompt, 
            model=self.domains_llm_model, 
            temperature=self.domains_llm_temperature, 
            context_data=context, 
            json_mode=True
        )
        try:
            json_response = json.loads(response)
            generated_list = json_response.get(list_type, [])
            return generated_list[:count]
        except (json.JSONDecodeError, KeyError) as e:
            print(f"{GREEN}System: {RESET}Error parsing LLM-generated {list_type} list: {e}")
            return []

    def get_domains(self, domains_input):
        """Generates or uses provided domains before the main run."""
        if isinstance(domains_input, int):
            domains_used = self._generate_list_from_llm("domains", domains_input, None)
        else:
            domains_used = domains_input
        return domains_used
    
    def get_instruction_prompt(self, domain):
        """
        Returns the core instruction prompt template for a given domain.
        """
        framework_template = self.config['framework_template'].replace('<num_concepts>', str(self.config['num_concepts']))
        return (
            self.config['abstraction_intro_template'] +
            self.config['string_domains_template'].replace("<domains>", domain) +
            framework_template +
            self.config['mapping_template'] +
            self.config['requirements_template']
        )
    
    def get_intelligent_context(self, instruction_prompt, original_query, context_files):
        """
        Inserts the instruction and query into the middle of the supplied context
        with empty lines above and below.
        """
        # Combine the context files into a single string
        full_context = "\n\n".join(context_files)
    
        # Create the new prompt string to be inserted
        insertion_text = (
            f"\n\n{instruction_prompt}\n\n"
            f"QUESTION:\n\n** {original_query} **\n\n"
        )
    
        # Find the midpoint of the full context string
        midpoint = len(full_context) // 2
    
        # Find a good place to split the string, such as a newline character,
        # to avoid splitting in the middle of a word or sentence
        split_point = full_context.find('\n', midpoint)
    
        # If a newline is not found after the midpoint, just split at the midpoint
        if split_point == -1:
            split_point = midpoint
    
        # Construct the final context by inserting the new text at the midpoint
        final_context = (
            full_context[:split_point] +
            insertion_text +
            full_context[split_point:]
        )
        return final_context

    def process_domain(self, original_query, domain, full_context=None):
        """
        Constructs and runs the prompt for a single domain using a pre-processed context.
        """
        print(f"{GREEN}System: {RESET}Processing domain '{YELLOW}{domain}{RESET}'.")
        
        instruction_prompt = self.get_instruction_prompt(domain)
        
        # Construct the final prompt, conditionally including the context.
        if full_context:
            final_prompt = f"INSTRUCTION PROMPT:\n\n{instruction_prompt}\n\n{original_query}\n\nCONTEXT:\n\n{full_context}\n\n** REMINDER **\n\nINSTRUCTION PROMPT:\n\n{instruction_prompt}\n\n{original_query}"
        else:
            final_prompt = f"INSTRUCTION PROMPT:\n\n{instruction_prompt}\n\n{original_query}"
        
        print(f"{GREEN}System: {RESET}Prompt for final_prompt:\n'{GREEN}{final_prompt}{RESET}'.")

        # Pass the pre-processed context to the LLM
        response, _ = self._call_ollama(
            final_prompt, 
            model=self.llm_model, 
            temperature=self.default_llm_temperature
        )
        return response

    def synthesize_wisdom(self, original_query, individual_responses):
        """Synthesizes all individual responses into a final, comprehensive answer."""
        print(f"\n{GREEN}System: {RESET}Synthesizing all individual responses into a final, comprehensive answer.")
        synthesis_prompt = self.config['synthesis_template'].format(
            responses='\n'.join(individual_responses),
            query=original_query
        )
        final_wisdom, _ = self._call_ollama(
            synthesis_prompt, 
            model=self.llm_model, 
            temperature=self.synthesis_llm_temperature
        )
        return final_wisdom, synthesis_prompt

    def run(self, original_query, domains_input, context_files=None):
        """Main method to orchestrate the VERA Protocol."""
        domains_used = self.get_domains(domains_input)
        
        individual_responses = []
        
        print(f"{GREEN}System: {RESET}Beginning iterative analysis of {len(domains_used)} domains.")
        
        for domain in domains_used:
            response = self.process_domain(original_query, domain, context_files)
            individual_responses.append(response)
            print(f"\n{YELLOW}Fragment from domain '{domain}':{RESET}\n{response}\n")

        final_wisdom, synthesis_prompt = self.synthesize_wisdom(original_query, individual_responses)
        
        return final_wisdom, domains_used, synthesis_prompt, individual_responses

