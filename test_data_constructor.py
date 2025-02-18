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

    def _format_data(self, data: Union[Dict, str], params: Dict[str, Any], context: Dict[str, Any] = None) -> Any:
        """
        Format data by replacing placeholders with actual values.
        
        Args:
            data: Dictionary or string to format
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
        field_value = jmespath.search(validation['field'], data)
        
        if validation['condition'] == 'not_empty':
            return field_value is not None and field_value != ''
        elif validation['condition'] == 'equals':
            return field_value == validation.get('value')
        elif validation['condition'] == 'contains':
            return validation.get('value') in field_value
        
        return False

    def _execute_step(self, step: Dict[str, Any], params: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """
        Execute a single aggregation step.
        
        Args:
            step: Step configuration
            params: User-provided parameters
            context: Context from previous steps
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

        # Apply JMESPath transformation
        if 'jmespath_query' in step:
            data = self._apply_jmespath(data, step['jmespath_query'])

        # Apply filter if specified
        if 'filter_query' in step:
            filter_query = self._format_data(step['filter_query'], params, context)
            data = self._apply_jmespath(data, filter_query)

        return data

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
        input_list = self._format_data(iteration_config['input_list'], params)
        input_param = iteration_config['input_param']
        
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
                    context[step['save_as']] = result
                
                # Validate results if validation is configured
                if 'validation' in iteration_config:
                    is_valid = self._validate_data(context, iteration_config['validation'])
                    if is_valid:
                        return context
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
            context = {'valid_items': valid_results}
        else:
            # Execute steps normally
            context = {}
            for step in aggregation['steps']:
                result = self._execute_step(step, params, context)
                context[step['save_as']] = result

        # Apply final transformation
        if 'transform_query' in aggregation:
            return self._apply_jmespath(context, aggregation['transform_query'])
        return context

    def save_test_data(self, data: Any, output_file: str):
        """Save the constructed test data to a JSON file."""
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

def main():
    # Example usage
    constructor = TestDataConstructor('aggregations.yaml')
    
    # Example 1: Find valid users from a list
    valid_users = constructor.construct_test_data(
        'AGG003',  # find_valid_users
        {
            'user_ids': ['12345', '67890', '11111', '22222']
        }
    )
    constructor.save_test_data(valid_users, 'valid_users.json')
    
    # Example 2: Get user order summary
    user_summary = constructor.construct_test_data(
        'AGG001',  # user_order_summary
        {'user_id': '12345'}
    )
    constructor.save_test_data(user_summary, 'user_order_summary.json')

if __name__ == '__main__':
    main() 