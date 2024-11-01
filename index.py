import re
from dataclasses import dataclass
from typing import List, Dict, Union
import numpy as np
import cv2
import pytesseract
from pdf2image import convert_from_path
import networkx as nx
from plantuml import PlantUML
import xml.etree.ElementTree as ET
import json

@dataclass
class UMLElement:
    id: str
    type: str  # class, interface, method, relationship, etc.
    name: str
    attributes: List[str] = None
    methods: List[str] = None
    relationships: List['UMLRelationship'] = None
    complexity: float = 1.0

@dataclass
class UMLRelationship:
    source: str
    target: str
    type: str  # inheritance, association, composition, etc.
    multiplicity: str = None

class UMLParser:
    def __init__(self):
        self.supported_types = {
            'class': self._parse_class_diagram,
            'sequence': self._parse_sequence_diagram,
            'activity': self._parse_activity_diagram,
            'component': self._parse_component_diagram
        }
        
        # Complexity weights for different UML elements
        self.complexity_weights = {
            'class': {
                'attributes': 0.2,
                'methods': 0.3,
                'relationships': 0.5
            },
            'sequence': {
                'actors': 0.2,
                'messages': 0.4,
                'conditions': 0.4
            },
            'activity': {
                'actions': 0.3,
                'decisions': 0.4,
                'parallel': 0.3
            },
            'component': {
                'interfaces': 0.3,
                'dependencies': 0.4,
                'components': 0.3
            }
        }

    def parse_uml(self, content: str, diagram_type: str) -> List[UMLElement]:
        """Parse UML content based on diagram type"""
        if diagram_type not in self.supported_types:
            raise ValueError(f"Unsupported UML diagram type: {diagram_type}")
            
        return self.supported_types[diagram_type](content)

    def _parse_class_diagram(self, content: str) -> List[UMLElement]:
        """Parse class diagram content"""
        elements = []
        current_class = None
        
        # Parse PlantUML-like syntax
        lines = content.split('\n')
        for line in lines:
            
            line = line.strip()
            
            # Class definition
            if line.startswith('class '):
                if current_class:
                    elements.append(current_class)
                class_name = line.split()[1]
                current_class = UMLElement(
                    id=len(elements),
                    type='class',
                    name=class_name,
                    attributes=[],
                    methods=[],
                    relationships=[]
                )
            
            # Attributes
            elif line.startswith('+') or line.startswith('-'):
                if current_class and ':' in line:
                    current_class.attributes.append(line)
            
            # Methods
            elif '()' in line:
                if current_class:
                    current_class.methods.append(line)
            
            # Relationships
            elif '-->' in line or '--*' in line or '--o' in line:
                parts = re.split(r'--|->|\*|o', line)
                if len(parts) >= 2:
                    relationship = UMLRelationship(
                        source=parts[0].strip(),
                        target=parts[-1].strip(),
                        type=self._determine_relationship_type(line)
                    )
                    if current_class:
                        current_class.relationships.append(relationship)
        
        # Add last class
        if current_class:
            elements.append(current_class)
        
        return elements

    def _parse_sequence_diagram(self, content: str) -> List[UMLElement]:
        """Parse sequence diagram content"""
        elements = []
        actors = set()
        messages = []
        
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            
            # Actor definition
            if line.startswith('participant '):
                actor = line.split()[1]
                actors.add(actor)
                elements.append(UMLElement(
                    id=len(elements),
                    type='actor',
                    name=actor
                ))
            
            # Message
            elif '->' in line:
                parts = line.split('->')
                if len(parts) == 2:
                    source = parts[0].strip()
                    target_message = parts[1].strip()
                    target = target_message.split(':')[0].strip()
                    message = target_message.split(':')[1].strip() if ':' in target_message else ''
                    
                    messages.append({
                        'source': source,
                        'target': target,
                        'message': message
                    })
        
        # Add messages as elements
        for msg in messages:
            elements.append(UMLElement(
                id=len(elements),
                type='message',
                name=msg['message'],
                attributes=[msg['source'], msg['target']]
            ))
        
        return elements

    def _parse_activity_diagram(self, content: str) -> List[UMLElement]:
        """Parse activity diagram content"""
        elements = []
        
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            
            # Start/End nodes
            if line.startswith('start') or line.startswith('end'):
                elements.append(UMLElement(
                    id=len(elements),
                    type='node',
                    name=line
                ))
            
            # Activities
            elif ':' in line and not any(symbol in line for symbol in ['->', '*', 'if', 'fork']):
                elements.append(UMLElement(
                    id=len(elements),
                    type='activity',
                    name=line.split(':')[1].strip()
                ))
            
            # Decision nodes
            elif 'if' in line:
                elements.append(UMLElement(
                    id=len(elements),
                    type='decision',
                    name=line
                ))
            
            # Parallel processing
            elif 'fork' in line or 'join' in line:
                elements.append(UMLElement(
                    id=len(elements),
                    type='parallel',
                    name=line
                ))
        
        return elements

    def _parse_component_diagram(self, content: str) -> List[UMLElement]:
        """Parse component diagram content"""
        elements = []
        
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            
            # Component definition
            if line.startswith('[') and ']' in line:
                name = line[1:line.index(']')]
                elements.append(UMLElement(
                    id=len(elements),
                    type='component',
                    name=name
                ))
            
            # Interface definition
            elif '(' in line and ')' in line:
                name = line[line.index('(') + 1:line.index(')')]
                elements.append(UMLElement(
                    id=len(elements),
                    type='interface',
                    name=name
                ))
            
            # Dependencies
            elif '-->' in line:
                parts = line.split('-->')
                if len(parts) == 2:
                    elements.append(UMLElement(
                        id=len(elements),
                        type='dependency',
                        name=f"{parts[0].strip()} -> {parts[1].strip()}"
                    ))
        
        return elements

    def _determine_relationship_type(self, line: str) -> str:
        """Determine the type of relationship from the line"""
        if '-->' in line:
            return 'association'
        elif '--*' in line:
            return 'composition'
        elif '--o' in line:
            return 'aggregation'
        elif '<|--' in line:
            return 'inheritance'
        else:
            return 'unknown'

class UMLEffortEstimator:
    def __init__(self):
        self.base_efforts = {
            'class': {
                'simple': 16,
                'medium': 24,
                'complex': 40
            },
            'method': {
                'simple': 8,
                'medium': 16,
                'complex': 24
            },
            'relationship': {
                'simple': 4,
                'medium': 8,
                'complex': 16
            },
            'interface': {
                'simple': 12,
                'medium': 20,
                'complex': 32
            }
        }
        
        self.complexity_factors = {
            'inheritance_depth': 1.2,
            'relationship_count': 1.1,
            'method_count': 1.15,
            'attribute_count': 1.1
        }

    def estimate_effort(self, elements: List[UMLElement]) -> Dict:
        """Calculate effort estimation based on UML elements"""
        total_effort = 0
        component_efforts = {}
        
        for element in elements:
            # Calculate base effort
            base_effort = self._calculate_base_effort(element)
            
            # Apply complexity factors
            complexity = self._calculate_complexity(element)
            
            # Calculate final effort
            element_effort = base_effort * complexity
            
            total_effort += element_effort
            component_efforts[element.name] = round(element_effort, 2)
        
        # Add buffer for testing and documentation (30%)
        total_effort *= 1.3
        
        return {
            'total_effort_hours': round(total_effort, 2),
            'component_efforts': component_efforts,
            'complexity_factors': {
                'high_complexity_elements': [
                    element.name for element in elements 
                    if self._calculate_complexity(element) > 1.5
                ]
            }
        }

    def _calculate_base_effort(self, element: UMLElement) -> float:
        """Calculate base effort for an element"""
        if element.type == 'class':
            if len(element.methods) > 10 or len(element.relationships) > 5:
                return self.base_efforts['class']['complex']
            elif len(element.methods) > 5 or len(element.relationships) > 3:
                return self.base_efforts['class']['medium']
            else:
                return self.base_efforts['class']['simple']
        
        # Add similar logic for other element types
        return self.base_efforts['method']['simple']

    def _calculate_complexity(self, element: UMLElement) -> float:
        """Calculate complexity factor for an element"""
        complexity = 1.0
        
        if element.type == 'class':
            # Inheritance depth
            if element.relationships:
                inheritance_count = sum(1 for r in element.relationships if r.type == 'inheritance')
                complexity *= (1 + (inheritance_count * (self.complexity_factors['inheritance_depth'] - 1)))
            
            # Relationship count
            relationship_count = len(element.relationships) if element.relationships else 0
            complexity *= (1 + (relationship_count * (self.complexity_factors['relationship_count'] - 1)))
            
            # Method count
            method_count = len(element.methods) if element.methods else 0
            complexity *= (1 + (method_count * (self.complexity_factors['method_count'] - 1)))
        
        return complexity

def analyze_uml_document(content: str, diagram_type: str) -> Dict:
    """Main function to analyze UML diagrams and estimate effort"""
    parser = UMLParser()
    estimator = UMLEffortEstimator()
    
    try:
        # Parse UML content
        elements = parser.parse_uml(content, diagram_type)
        
        # Calculate effort estimation
        estimation = estimator.estimate_effort(elements)
        
        # Generate detailed report
        report = {
            'summary': {
                'total_effort_hours': estimation['total_effort_hours'],
                'calendar_days': int(np.ceil(estimation['total_effort_hours'] / 6)),
                'number_of_components': len(elements)
            },
            'component_breakdown': estimation['component_efforts'],
            'complexity_analysis': estimation['complexity_factors'],
            'recommendations': []
        }
        
        # Add recommendations
        if len(estimation['complexity_factors']['high_complexity_elements']) > 0:
            report['recommendations'].append(
                "Consider breaking down complex elements into smaller components"
            )
        
        return report
        
    except Exception as e:
        return {
            'error': f"Error processing UML diagram: {str(e)}",
            'status': 'failed'
        }

# Example usage
def main():
    # Example UML class diagram
    class_diagram = """
    class User {
        -id: int
        -name: string
        +getId(): int
        +setName(name: string): void
    }
    
    class Order {
        -orderId: int
        -items: List
        +addItem(item: Item): void
        +removeItem(item: Item): void
    }
    
    User --> Order
    """
    
    # Generate estimation for class diagram
    print("\nAnalyzing Class Diagram")
    report = analyze_uml_document(class_diagram, 'class')
    
    if 'error' in report:
        print(f"Error: {report['error']}")
    else:
        print("\nEffort Estimation Report")
        print("=======================")
        print(f"Total Effort: {report['summary']['total_effort_hours']} hours")
        print(f"Calendar Days: {report['summary']['calendar_days']} days")
        print(f"Number of Components: {report['summary']['number_of_components']}")
        
        print("\nComponent Breakdown:")
        for component, effort in report['component_breakdown'].items():
            print(f"- {component}: {effort} hours")
        
        print("\nRecommendations:")
        for rec in report['recommendations']:
            print(f"- {rec}")

if __name__ == "__main__":
    main()