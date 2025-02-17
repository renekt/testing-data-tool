# Test Data Constructor

A Python tool for constructing test data by calling downstream APIs and transforming their responses using JMESPath queries. This implementation uses OpenAPI specifications for API definitions and supports complex data aggregations.

## Features

- API definitions using OpenAPI 3.0 specifications
- Flexible data aggregation configurations
- Support for multiple downstream services
- JMESPath-based data transformation and filtering
- Environment variable support for API keys
- Caching of API specifications

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── apis/                    # OpenAPI specifications
│   ├── pacs_api.yaml       # PACS service API spec
│   └── product_api.yaml    # Product service API spec
├── aggregations.yaml       # Data aggregation configurations
├── test_data_constructor.py # Main implementation
└── requirements.txt        # Python dependencies
```

## Configuration

1. Create a `.env` file with your API credentials (if required):
```
API_KEY=your_api_key_here
```

2. Define your APIs using OpenAPI 3.0 specifications in the `apis/` directory:
```yaml
openapi: 3.0.0
info:
  title: API Name
  version: 1.0.0
servers:
  - url: https://api.example.com
paths:
  /path/{param}:
    get:
      operationId: uniqueOperationId
      parameters:
        - name: param
          in: path
          required: true
          schema:
            type: string
      responses:
        '200':
          description: Success response
```

3. Configure your data aggregations in `aggregations.yaml`:
```yaml
version: 1.0.0
aggregations:
  example_aggregation:
    id: AGG001
    description: "Description of the aggregation"
    steps:
      - id: step1
        operation_id: uniqueOperationId
        api_spec: apis/service_api.yaml
        params:
          path:
            param: "{value}"
        save_as: result_key
    transform_query: "jmespath_expression"
```

## Usage

```python
from test_data_constructor import TestDataConstructor

# Initialize the constructor
constructor = TestDataConstructor('aggregations.yaml')

# Construct test data using an aggregation
data = constructor.construct_test_data(
    'AGG001',  # Aggregation ID
    {'param': 'value'}  # Parameters
)

# Save the test data
constructor.save_test_data(data, 'output.json')
```

## Aggregation Configuration

Each aggregation consists of:

1. **ID**: Unique identifier for the aggregation
2. **Description**: Human-readable description
3. **Steps**: Sequence of API calls and transformations
   - `operation_id`: References an operation in the OpenAPI spec
   - `api_spec`: Path to the OpenAPI specification file
   - `params`: Parameters for the API call (path, query, body)
   - `save_as`: Key to store the result in the context
   - `jmespath_query`: Optional transformation query
   - `filter_query`: Optional filter query
4. **Transform Query**: Final JMESPath transformation

## JMESPath Query Examples

1. Extract specific fields:
```yaml
jmespath_query: "data.{id: id, name: name, email: email}"
```

2. Filter and transform arrays:
```yaml
jmespath_query: "items[?status=='active'].{id: id, name: name}"
```

3. Reference previous results:
```yaml
body:
  user_id: "{user_info.id}"
  items: "{orders[*].product_id}"
```

## Error Handling

The tool includes error handling for:
- Invalid OpenAPI specifications
- Missing operations
- API request failures
- Invalid JMESPath queries
- Missing parameters 