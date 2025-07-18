
import React, { useState } from 'react';
import { Key, Eye, EyeOff } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';

interface ApiKeyInputProps {
  onApiKeySet: (apiKey: string) => void;
  apiKey: string;
}

const ApiKeyInput: React.FC<ApiKeyInputProps> = ({ onApiKeySet, apiKey }) => {
  const [localApiKey, setLocalApiKey] = useState(apiKey);
  const [showApiKey, setShowApiKey] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (localApiKey.trim()) {
      onApiKeySet(localApiKey.trim());
    }
  };

  return (
    <Card className="w-full max-w-md mx-auto">
      <CardHeader className="text-center">
        <div className="w-12 h-12 bg-gradient-to-r from-purple-600 to-purple-700 rounded-full flex items-center justify-center mx-auto mb-2">
          <Key className="w-6 h-6 text-white" />
        </div>
        <CardTitle className="text-xl font-semibold">Backend Configuration</CardTitle>
        <CardDescription>
          Enter your backend API endpoint or API key to connect the chatbot
        </CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="relative">
            <Input
              type={showApiKey ? "text" : "password"}
              value={localApiKey}
              onChange={(e) => setLocalApiKey(e.target.value)}
              placeholder="Enter your backend API key or endpoint..."
              className="pr-10"
            />
            <Button
              type="button"
              variant="ghost"
              size="sm"
              className="absolute right-0 top-0 h-full px-3"
              onClick={() => setShowApiKey(!showApiKey)}
            >
              {showApiKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
            </Button>
          </div>
          <Button 
            type="submit" 
            className="w-full bg-gradient-to-r from-purple-600 to-purple-700 hover:from-purple-700 hover:to-purple-800"
            disabled={!localApiKey.trim()}
          >
            Connect Backend
          </Button>
        </form>
        <p className="text-xs text-gray-500 mt-4 text-center">
          Your API key is stored locally and never sent to external servers
        </p>
      </CardContent>
    </Card>
  );
};

export default ApiKeyInput;
