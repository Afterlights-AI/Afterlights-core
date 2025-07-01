from openai import OpenAI

class ModelContext:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.history = []
        self.client = OpenAI()
    def get_history(self):
        """Return the conversation history."""
        return self.history
    
    def __repr__(self):
        return f"ModelContext(model_name={self.model_name}, model_version={self.model_version})"

    def __str__(self):
        return f"{self.model_name} v{self.model_version}"
    
    def add_system_message(self, messages: list):
        """Add a system message to the conversation history."""
        self.history.append({"role": "system", "content": messages})    
    
    def add_user_message(self, messages: list):
        """Add a user message to the conversation history."""
        self.history.append({"role": "user", "content": messages})
    
    def add_assistant_message(self, messages: list):
        """Add an assistant message to the conversation history."""
        self.history.append({"role": "assistant", "content": messages})
    
    def add_function_call(self, fn_call):
        self.history.append(fn_call)
                                          
    
    def add_tool_message(self, *,
            tool_call_id: str, 
            tool_result: any = None):
        """Add a tool message to the conversation history."""
        self.history.append({
            "type": "function_call_output",
            "call_id": tool_call_id,
            "output": str(tool_result)
        })
    
    def clear_history(self):
        """Clear the conversation history."""
        self.history = []
    
    def get_last_message(self):
        """Return the last message in the conversation history."""
        if self.history:
            return self.history[-1]
        return None
    
    def get_last_user_message(self):
        """Return the last user message in the conversation history."""
        for msg in reversed(self.history):
            if msg['role'] == 'user':
                return msg
        return None
    
    def call_model_with_tools(self, TOOLS: list):
        """Call the model with the current conversation history."""
        response = self.client.responses.create(
            model=self.model_name,
            input=self.history,
            tools=TOOLS,
            tool_choice="auto"  # let the model decide
        )
        print("resp output", response.output[0])
        return response.output[0]
    