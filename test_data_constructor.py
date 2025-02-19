#!/usr/bin/env python3
import os
import json
import yaml
import jmespath
import requests
from typing import Dict, Any, Optional, Union, List
from dotenv import load_dotenv
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from pathlib import Path
import sys

class OpenAPISpec:
    def __init__(self, spec_file: str):
        """
        Load and parse an OpenAPI specification file.
        
        Args:
            spec_file (str): Path to the OpenAPI specification file
        """
        with open(spec_file, 'r') as f:
            self.spec = yaml.safe_load(f)
        self.operation_map = self._build_operation_map()

    def _build_operation_map(self) -> Dict[str, Dict[str, Any]]:
        """Build a map of operationId to path and method."""
        operation_map = {}
        for path, methods in self.spec['paths'].items():
            for method, operation in methods.items():
                if 'operationId' in operation:
                    operation_map[operation['operationId']] = {
                        'path': path,
                        'method': method.upper(),
                        'operation': operation
                    }
        return operation_map

    def get_operation(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """Get operation details by operationId."""
        return self.operation_map.get(operation_id)

    def get_base_url(self) -> str:
        """Get the base URL from the servers list."""
        return self.spec['servers'][0]['url']

class TestDataConstructor:
    def __init__(self, aggregation_file: str, max_workers: int = 5):
        """
        Initialize the TestDataConstructor with an aggregation configuration file.
        
        Args:
            aggregation_file (str): Path to the aggregation configuration file
            max_workers (int): Maximum number of concurrent workers for parallel processing
        """
        self.aggregations = self._load_config(aggregation_file)
        self.api_specs = {}  # Cache for loaded OpenAPI specs
        self.max_workers = max_workers
        load_dotenv()

    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load and parse the aggregation configuration file."""
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)

    def _get_api_spec(self, spec_file: str) -> OpenAPISpec:
        """Get or load an OpenAPI specification."""
        if spec_file not in self.api_specs:
            self.api_specs[spec_file] = OpenAPISpec(spec_file)
        return self.api_specs[spec_file]

    def _format_path(self, path: str, params: Dict[str, Any]) -> str:
        """Format path parameters in the URL path."""
        for key, value in params.items():
            path = path.replace(f"{{{key}}}", str(value))
        return path

    def _format_data(self, data: Union[Dict, str, List], params: Dict[str, Any], context: Dict[str, Any] = None) -> Any:
        """
        Format data by replacing placeholders with actual values.
        
        Args:
            data: Dictionary, string, or list to format
            params: User-provided parameters
            context: Context from previous steps
        """
        if isinstance(data, str):
            # First try to evaluate JMESPath expressions in the context
            if context and data.startswith("{") and data.endswith("}"):
                expr = data[1:-1]
                try:
                    result = jmespath.search(expr, context)
                    if result is not None:
                        return result
                except:
                    pass
            # Then try standard string formatting
            return data.format(**params) if params else data
        elif isinstance(data, dict):
            return {k: self._format_data(v, params, context) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._format_data(item, params, context) for item in data]
        return data

    def _validate_data(self, data: Dict[str, Any], validation: Dict[str, Any]) -> bool:
        """
        Validate data according to validation rules.
        
        Args:
            data: Data to validate
            validation: Validation configuration
        
        Returns:
            bool: Whether the data is valid
        """
        try:
            field_value = jmespath.search(validation['field'], data)
            print(f"Validating field '{validation['field']}' with value: {field_value}")
            
            if field_value is None:
                print(f"Warning: Field '{validation['field']}' not found in data")
                return False
                
            if validation['condition'] == 'not_empty':
                return field_value is not None and str(field_value).strip() != ''
            elif validation['condition'] == 'equals':
                return str(field_value) == str(validation.get('value', ''))
            elif validation['condition'] == 'contains':
                target_value = str(validation.get('value', ''))
                field_str = str(field_value)
                result = target_value in field_str
                print(f"Checking if '{target_value}' is in '{field_str}': {result}")
                return result
            
            print(f"Warning: Unknown validation condition '{validation['condition']}'")
            return False
            
        except Exception as e:
            print(f"Error during validation: {str(e)}")
            return False

    def _execute_step(self, step: Dict[str, Any], params: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """
        Execute a single aggregation step.
        
        Args:
            step: Step configuration
            params: User-provided parameters
            context: Context from previous steps
            
        Returns:
            Any: Step result with validation information
        """
        # Load API specification
        api_spec = self._get_api_spec(step['api_spec'])
        operation = api_spec.get_operation(step['operation_id'])
        if not operation:
            raise ValueError(f"Operation {step['operation_id']} not found in {step['api_spec']}")

        # Prepare request
        base_url = api_spec.get_base_url()
        path = operation['path']
        
        # Handle path parameters
        if 'path' in step.get('params', {}):
            path_params = self._format_data(step['params']['path'], params, context)
            path = self._format_path(path, path_params)

        url = urljoin(base_url, path)

        # Prepare headers
        headers = {'Content-Type': 'application/json'}
        api_key = os.getenv('API_KEY')
        if api_key:
            headers['Authorization'] = f"Bearer {api_key}"

        # Prepare query parameters
        query_params = {}
        if 'query' in step.get('params', {}):
            query_params = self._format_data(step['params']['query'], params, context)

        # Prepare request body
        body = None
        if 'body' in step.get('params', {}):
            body = self._format_data(step['params']['body'], params, context)
        
        print(f"Making request to {url} with method {operation['method']}")
        print(f"Query params: {query_params}")
        print(f"Request body: {body}")
        
        # Make request
        response = requests.request(
            method=operation['method'],
            url=url,
            headers=headers,
            params=query_params,
            json=body
        )
        response.raise_for_status()
        data = response.json()
        print(f"Response status code: {response.status_code}")
        print(f"Response data: {data}")

        # Apply JMESPath transformation
        if 'jmespath_query' in step:
            data = self._apply_jmespath(data, step['jmespath_query'])
            print(f"After JMESPath transformation: {data}")

        # Apply filter if specified
        if 'filter_query' in step:
            filter_query = self._format_data(step['filter_query'], params, context)
            data = self._apply_jmespath(data, filter_query)

        # Apply step-level validation if specified
        result = {
            'data': data,
            'is_valid': True,
            'validation_result': None
        }
        
        if 'validation' in step:
            # Use data directly instead of creating a new context
            is_valid = self._validate_data({'data': data}, {'field': 'data.' + step['validation']['field'], **step['validation']})
            result['is_valid'] = is_valid
            result['validation_result'] = {
                'field': step['validation']['field'],
                'condition': step['validation']['condition'],
                'value': step['validation']['value'],
                'passed': is_valid
            }
            print(f"Step {step['id']} validation {'passed' if is_valid else 'failed'}")

        return result

    def _execute_iteration(self, aggregation: Dict[str, Any], params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Execute steps for each item in the input list.
        
        Args:
            aggregation: Aggregation configuration
            params: Parameters including the input list
        
        Returns:
            List[Dict[str, Any]]: List of results for valid items
        """
        iteration_config = aggregation['iteration']
        input_list_key = iteration_config['input_list'][1:-1]  # Remove the curly braces
        input_list = params.get(input_list_key)
        
        if not isinstance(input_list, list):
            raise ValueError(f"Input list '{input_list_key}' must be an array, got {type(input_list)}")
        
        if not input_list:
            raise ValueError(f"Input list '{input_list_key}' cannot be empty")
            
        input_param = iteration_config['input_param']
        print(f"Processing {len(input_list)} items with parameter '{input_param}'")
        
        all_results = []
        valid_results = []

        def process_item(item):
            # Prepare parameters for this iteration
            item_params = {**params, input_param: item}
            
            try:
                # Execute steps for this item
                context = {}
                for step in aggregation['steps']:
                    result = self._execute_step(step, item_params, context)
                    context[step['save_as']] = result['data']
                    context[f"{step['save_as']}_validation"] = {
                        'is_valid': result['is_valid'],
                        'validation_result': result['validation_result']
                    }
                    all_results.append({
                        'step_id': step['id'],
                        'data': result['data'],
                        'validation': result['validation_result']
                    })
                
                # Validate results if validation is configured
                if 'validation' in iteration_config:
                    is_valid = self._validate_data(context, iteration_config['validation'])
                    if is_valid:
                        print(f"Item {item} is valid")
                        return context
                    else:
                        print(f"Item {item} failed validation")
                else:
                    return context
            except Exception as e:
                print(f"Error processing item {item}: {str(e)}")
            
            return None

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_item = {executor.submit(process_item, item): item for item in input_list}
            for future in as_completed(future_to_item):
                result = future.result()
                if result is not None:
                    valid_results.append(result)

        print(f"Found {len(valid_results)} valid items out of {len(input_list)} total items")
        return valid_results

    def _apply_jmespath(self, data: Any, query: str) -> Any:
        """Apply JMESPath query to extract and transform data."""
        return jmespath.search(query, data)

    def construct_test_data(self, aggregation_id: str, params: Dict[str, Any]) -> Any:
        """
        Construct test data according to the aggregation configuration.
        
        Args:
            aggregation_id (str): ID of the aggregation to execute
            params (Dict[str, Any]): Parameters for the aggregation
        
        Returns:
            Any: Aggregated and transformed test data
        """
        # Find aggregation configuration
        aggregation = None
        for agg in self.aggregations['aggregations'].values():
            if agg['id'] == aggregation_id:
                aggregation = agg
                break

        if not aggregation:
            raise ValueError(f"Aggregation with ID {aggregation_id} not found")

        # Handle iteration if configured
        if 'iteration' in aggregation:
            valid_results = self._execute_iteration(aggregation, params)
            context = {
                'valid_items': valid_results,
                'params': params  # Add params to context
            }
        else:
            # Execute steps normally
            context = {'params': params}  # Initialize context with params
            all_results = []
            for step in aggregation['steps']:
                result = self._execute_step(step, params, context)
                if result is None:  # Handle case where step execution failed completely
                    result = {
                        'data': None,
                        'is_valid': False,
                        'validation_result': None
                    }
                
                # Store both the data and validation result in context
                context[step['save_as']] = result['data']
                context[f"{step['save_as']}_validation"] = {
                    'is_valid': result['is_valid'],
                    'validation_result': result['validation_result']
                }
                all_results.append({
                    'step_id': step['id'],
                    'data': result['data'],
                    'validation': result['validation_result']
                })
            
            # Add all results to context
            context['all_results'] = all_results
            # Safely handle validation results
            context['valid_results'] = [
                r for r in all_results 
                if r and r.get('validation') and r['validation'].get('passed', False)
            ]

        # Apply final transformation
        if 'transform_query' in aggregation:
            return self._apply_jmespath(context, aggregation['transform_query'])
        return context

    def save_test_data(self, data: Any, output_file: str):
        """Save the constructed test data to a JSON file."""
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

class TestScenario:
    """Predefined test scenarios for easy testing"""
    
    @staticmethod
    def find_valid_users() -> Dict[str, Any]:
        """Test scenario for finding valid users"""
        return {
            'aggregation_id': 'AGG003',
            'params': {
                'user_ids': ['12345', '67890', '11111', '22222']
            },
            'output_file': 'valid_users.json'
        }
    
    @staticmethod
    def user_order_summary() -> Dict[str, Any]:
        """Test scenario for getting user order summary"""
        return {
            'aggregation_id': 'AGG001',
            'params': {
                'user_id': '12345'
            },
            'output_file': 'user_order_summary.json'
        }
    
    @staticmethod
    def product_order_details() -> Dict[str, Any]:
        """Test scenario for getting product order details"""
        return {
            'aggregation_id': 'AGG002',
            'params': {
                'user_id': '12345',
                'order_id': 'ORDER789'
            },
            'output_file': 'order_details.json'
        }

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Test Data Constructor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  1. Run with predefined scenario:
     python test_data_constructor.py --scenario find_valid_users

  2. Run with custom parameters (using single quotes):
     python test_data_constructor.py --aggregation-id AGG003 --params '{"user_ids": ["12345", "67890"]}'

  3. Run with custom parameters (using escaped double quotes):
     python test_data_constructor.py --aggregation-id AGG003 --params "{\"user_ids\": [\"12345\", \"67890\"]}"

  4. Run with parameters from file:
     python test_data_constructor.py --aggregation-id AGG003 --params-file params.json
'''
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='aggregations.yaml',
        help='Path to aggregation configuration file'
    )
    
    parser.add_argument(
        '--scenario',
        type=str,
        choices=['find_valid_users', 'user_order_summary', 'product_order_details'],
        help='Predefined test scenario to run'
    )
    
    parser.add_argument(
        '--aggregation-id',
        type=str,
        help='ID of the aggregation to execute'
    )
    
    # 参数组：params 和 params-file 互斥
    params_group = parser.add_mutually_exclusive_group()
    params_group.add_argument(
        '--params',
        type=str,  # Changed from json.loads to str to handle parsing separately
        help='JSON string of parameters for the aggregation'
    )
    params_group.add_argument(
        '--params-file',
        type=str,
        help='Path to JSON file containing parameters'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output file path for the test data'
    )
    
    args = parser.parse_args()
    
    # Handle parameter parsing
    if args.params:
        try:
            args.params = json.loads(args.params)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in --params: {str(e)}\nMake sure to properly quote or escape the JSON string.")
    elif args.params_file:
        try:
            with open(args.params_file, 'r') as f:
                args.params = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            raise ValueError(f"Error reading parameters from {args.params_file}: {str(e)}")
    
    return args

def main():
    """Main function that can be run from PyCharm or command line"""
    try:
        args = parse_args()
        
        # Initialize constructor
        constructor = TestDataConstructor(args.config)
        
        # If scenario is specified, use predefined test scenario
        if args.scenario:
            scenario = getattr(TestScenario, args.scenario)()
            aggregation_id = scenario['aggregation_id']
            params = scenario['params']
            output_file = scenario['output_file']
        else:
            # Use command line arguments
            if not args.aggregation_id or (not args.params and not args.params_file):
                raise ValueError("Either --scenario or both --aggregation-id and (--params or --params-file) must be specified")
            aggregation_id = args.aggregation_id
            params = args.params
            output_file = args.output or f"{aggregation_id}_output.json"
        
        # Execute the aggregation
        print(f"Executing aggregation {aggregation_id} with params: {json.dumps(params, indent=2)}")
        result = constructor.construct_test_data(aggregation_id, params)
        
        # Save results
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        constructor.save_test_data(result, output_file)
        print(f"Results saved to {output_file}")
        
        # Print summary
        if isinstance(result, dict):
            print("\nResult Summary:")
            for key, value in result.items():
                if isinstance(value, list):
                    print(f"{key}: {len(value)} items")
                else:
                    print(f"{key}: {value}")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == '__main__':
    main() 