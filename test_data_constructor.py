#!/usr/bin/env python3
import os
import json
import yaml
import jmespath
import requests
from typing import Dict, Any, Optional, Union, List
from dotenv import load_dotenv
from urllib.parse import urljoin

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
    def __init__(self, aggregation_file: str):
        """
        Initialize the TestDataConstructor with an aggregation configuration file.
        
        Args:
            aggregation_file (str): Path to the aggregation configuration file
        """
        self.aggregations = self._load_config(aggregation_file)
        self.api_specs = {}  # Cache for loaded OpenAPI specs
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

        # Execute steps
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
    
    # Example 1: Get user order summary
    user_summary = constructor.construct_test_data(
        'AGG001',  # user_order_summary
        {'user_id': '12345'}
    )
    constructor.save_test_data(user_summary, 'user_order_summary.json')
    
    # Example 2: Get product order details
    order_details = constructor.construct_test_data(
        'AGG002',  # product_order_details
        {
            'user_id': '12345',
            'order_id': 'ORDER789'
        }
    )
    constructor.save_test_data(order_details, 'order_details.json')

if __name__ == '__main__':
    main() 